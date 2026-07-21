"""Training callbacks for DROID: epoch metrics/plots, Slack notifications, and
end-of-training validation-set inference. Kept separate from the training recipe
and model description modules so those files only need to wire these in."""

import io
import json
import logging
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tensorflow as tf

from ml4h.metrics import coefficient_of_determination, get_precision_recall_aucs, get_roc_aucs, survival_likelihood_loss

METRICS_SUBDIR = 'metrics'
VALIDATION_INFERENCE_SUBDIR = 'validation_inference'
SLACK_WEBHOOK_ENV_VAR = 'SLACK_WEBHOOK_URL'


def _write_bytes(path, data):
    with tf.io.gfile.GFile(path, 'wb') as f:
        f.write(data)


def _write_text(path, text):
    with tf.io.gfile.GFile(path, 'w') as f:
        f.write(text)


def _savefig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()


class MetricsHistoryCallback(tf.keras.callbacks.Callback):
    """Writes a CSV and PNG plots of per-epoch training/validation loss and metrics.

    Reads whatever keys Keras puts in the epoch `logs` dict rather than assuming a
    fixed set of output heads or metric names, so it keeps working as
    --output_labels/--survival_task configuration changes.
    """

    def __init__(self, output_folder):
        super().__init__()
        self.output_folder = output_folder.rstrip('/')
        self.metrics_folder = f'{self.output_folder}/{METRICS_SUBDIR}'
        self.csv_path = f'{self.metrics_folder}/training_history.csv'
        self.loss_plot_path = f'{self.metrics_folder}/loss_curves.png'
        self.metric_plot_path = f'{self.metrics_folder}/metric_curves.png'
        self.history = []
        tf.io.gfile.makedirs(self.metrics_folder)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = {'epoch': int(epoch)}
        for key, value in logs.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
        self.history.append(row)

        df = pd.DataFrame(self.history).set_index('epoch')
        _write_text(self.csv_path, df.to_csv())

        try:
            self._plot_loss_curves(df)
            self._plot_metric_curves(df)
        except Exception:
            logging.exception('Failed to render training curve plots for epoch %d.', epoch)

    def _plot_curves(self, df, columns, path):
        columns = sorted(columns)
        if not columns:
            return
        fig, axes = plt.subplots(len(columns), 1, figsize=(8, 3 * len(columns)), squeeze=False)
        for ax, column in zip(axes[:, 0], columns):
            ax.plot(df.index, df[column], label='train')
            val_column = f'val_{column}'
            if val_column in df.columns:
                ax.plot(df.index, df[val_column], label='validation')
            ax.set_title(column)
            ax.set_xlabel('epoch')
            ax.legend()
        fig.tight_layout()
        _write_bytes(path, _savefig_bytes(fig))

    def _plot_loss_curves(self, df):
        loss_columns = [
            c for c in df.columns
            if not c.startswith('val_') and (c == 'loss' or c.endswith('_loss'))
        ]
        self._plot_curves(df, loss_columns, self.loss_plot_path)

    def _plot_metric_curves(self, df):
        metric_columns = [
            c for c in df.columns
            if not c.startswith('val_') and c != 'loss' and not c.endswith('_loss')
        ]
        self._plot_curves(df, metric_columns, self.metric_plot_path)


class SlackNotifierCallback(tf.keras.callbacks.Callback):
    """Sends text-only Slack messages via an incoming webhook on training start,
    training success, training failure, and at the end of each epoch.

    The webhook URL is read from the SLACK_WEBHOOK_URL environment variable (no
    CLI flag) so notifications are on by default and silently disable themselves,
    with a single warning, when no webhook is configured.
    """

    def __init__(self, output_folder, run_summary=None, timeout=10):
        super().__init__()
        self.output_folder = output_folder
        self.run_summary = run_summary or {}
        self.timeout = timeout
        self.webhook_url = os.environ.get(SLACK_WEBHOOK_ENV_VAR)
        if not self.webhook_url:
            logging.warning(
                '%s is not set; Slack training notifications are disabled.', SLACK_WEBHOOK_ENV_VAR,
            )

    def _send(self, text):
        if not self.webhook_url:
            return
        try:
            response = requests.post(self.webhook_url, json={'text': text}, timeout=self.timeout)
            response.raise_for_status()
        except Exception:
            logging.exception('Failed to send Slack notification.')

    def on_train_begin(self, logs=None):
        lines = ['DROID training started.', f'Output folder: {self.output_folder}']
        for key, value in self.run_summary.items():
            lines.append(f'{key}: {value}')
        self._send('\n'.join(lines))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metric_lines = '\n'.join(f'{key}: {value:.4f}' for key, value in sorted(logs.items()))
        self._send(f'DROID training - epoch {epoch} complete.\nOutput folder: {self.output_folder}\n{metric_lines}')

    def notify_failure(self, exception):
        self._send(
            f'DROID training FAILED.\nOutput folder: {self.output_folder}\nError: {exception}',
        )

    def notify_success(self, validation_summary=None):
        lines = ['DROID training completed successfully.', f'Output folder: {self.output_folder}']
        if validation_summary:
            lines.append('Validation performance (best checkpoint):')
            for target, target_metrics in validation_summary.items():
                lines.append(f'  {target}:')
                for metric_name, value in target_metrics.items():
                    lines.append(f'    {metric_name}: {value}')
        self._send('\n'.join(lines))


def _flatten_metrics_report(metrics_report):
    rows = []
    for category, targets in metrics_report.items():
        for target, target_metrics in targets.items():
            for metric_name, value in target_metrics.items():
                if isinstance(value, dict):
                    for sub_label, sub_value in value.items():
                        rows.append({
                            'category': category, 'target': target,
                            'metric': f'{metric_name}_{sub_label}', 'value': sub_value,
                        })
                else:
                    rows.append({'category': category, 'target': target, 'metric': metric_name, 'value': value})
    return pd.DataFrame(rows)


def run_validation_inference(
        model,
        valid_dataset,
        n_valid_steps,
        batch_size,
        valid_ids,
        output_folder,
        output_labels,
        output_reg_len,
        cls_category_map_dicts,
        survival_tasks,
):
    """Runs a single inference pass over the validation set with the model's current
    (best-checkpoint) weights, computes per-target validation metrics, and saves both
    the raw predictions and the metrics report under output_folder/validation_inference.

    Returns a summary dict of {target_name: {metric_name: value}} for reporting (e.g. via Slack).
    """
    output_folder = output_folder.rstrip('/')
    inference_folder = f'{output_folder}/{VALIDATION_INFERENCE_SUBDIR}'
    tf.io.gfile.makedirs(inference_folder)

    cls_output_names = (cls_category_map_dicts or {}).get('cls_output_order', [])
    output_names = list(model.output_names)

    y_true_batches = {name: [] for name in output_names}
    y_pred_batches = {name: [] for name in output_names}

    for batch_inputs, batch_outputs in valid_dataset.take(n_valid_steps).as_numpy_iterator():
        batch_preds = model.predict_on_batch(batch_inputs)
        if not isinstance(batch_preds, (list, tuple)):
            batch_preds = [batch_preds]
        if not isinstance(batch_outputs, (list, tuple)):
            batch_outputs = [batch_outputs]
        for name, true_vals, pred_vals in zip(output_names, batch_outputs, batch_preds):
            y_true_batches[name].append(true_vals)
            y_pred_batches[name].append(pred_vals)

    y_true = {name: np.concatenate(vals, axis=0) for name, vals in y_true_batches.items() if vals}
    y_pred = {name: np.concatenate(vals, axis=0) for name, vals in y_pred_batches.items() if vals}

    metrics_report = {}

    if output_reg_len > 0 and 'echolab' in y_true:
        reg_labels = output_labels[:output_reg_len]
        reg_true, reg_pred = y_true['echolab'], y_pred['echolab']
        reg_metrics = {}
        for i, label in enumerate(reg_labels):
            errors = reg_true[:, i] - reg_pred[:, i]
            reg_metrics[label] = {
                'mae': float(np.mean(np.abs(errors))),
                'rmse': float(np.sqrt(np.mean(errors ** 2))),
                'r2': float(coefficient_of_determination(reg_true[:, i], reg_pred[:, i])),
            }
        metrics_report['regression'] = reg_metrics

    cls_metrics = {}
    for cls_name in cls_output_names:
        output_name = f'cls_{cls_name}'
        if output_name not in y_true:
            continue
        cls_true, cls_pred = y_true[output_name], y_pred[output_name]
        # Coerce keys to str: raw category values may be numpy scalars (e.g. np.int64 for
        # integer-coded categories), which json.dumps rejects as dict keys further below.
        label_map = {str(k): v for k, v in cls_category_map_dicts[cls_name].items()}
        roc_aucs = get_roc_aucs({'model': cls_pred}, cls_true, label_map)['model']
        pr_aucs = get_precision_recall_aucs({'model': cls_pred}, cls_true, label_map)['model']
        cls_metrics[cls_name] = {
            'accuracy': float(np.mean(cls_true.argmax(axis=1) == cls_pred.argmax(axis=1))),
            'auroc_macro': float(np.mean(list(roc_aucs.values()))),
            'auprc_macro': float(np.mean(list(pr_aucs.values()))),
            'auroc_per_class': roc_aucs,
            'auprc_per_class': pr_aucs,
        }
    if cls_metrics:
        metrics_report['classification'] = cls_metrics

    survival_metrics = {}
    for task in (survival_tasks or []):
        output_name = f"survival_{task['name']}"
        if output_name not in y_true:
            continue
        loss_fn = survival_likelihood_loss(task['intervals'])
        per_sample_loss = loss_fn(
            tf.constant(y_true[output_name]), tf.constant(y_pred[output_name]),
        ).numpy()
        survival_metrics[task['name']] = {'mean_negative_log_likelihood': float(np.mean(per_sample_loss))}
    if survival_metrics:
        metrics_report['survival'] = survival_metrics

    predictions_df = pd.DataFrame({'sample_id': valid_ids[:n_valid_steps * batch_size]})
    for name in y_true:
        # y_true and y_pred can differ in width (e.g. survival labels are 2*intervals wide —
        # survived/failure flags — while the model's survival head only outputs `intervals`
        # columns), so each is indexed against its own shape rather than a shared range.
        for i in range(y_true[name].shape[1]):
            predictions_df[f'{name}_true_{i}'] = y_true[name][:, i]
        for i in range(y_pred[name].shape[1]):
            predictions_df[f'{name}_pred_{i}'] = y_pred[name][:, i]

    parquet_buffer = io.BytesIO()
    predictions_df.to_parquet(parquet_buffer)
    _write_bytes(f'{inference_folder}/predictions.pq', parquet_buffer.getvalue())

    _write_text(f'{inference_folder}/metrics.json', json.dumps(metrics_report, indent=2))
    _write_text(f'{inference_folder}/metrics.csv', _flatten_metrics_report(metrics_report).to_csv(index=False))

    summary = {}
    for category, targets in metrics_report.items():
        for target, target_metrics in targets.items():
            summary[f'{category}/{target}'] = {
                k: v for k, v in target_metrics.items() if not isinstance(v, dict)
            }
    return summary
