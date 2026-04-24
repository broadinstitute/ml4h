# Make a deployment docker with a model from the Model Factory
Edit `Dockerfile` to copy and load your `.keras` model file.
Then build the docker image:
```bash 
docker build -t multimodal_finngen_deploy .
```
Then run the docker image:  
For multimodal inference:  
```bash 
docker run --rm -v "~/ecg_xml:/work" multimodal_finngen_deploy  --mode infer --path /work/xmls --metadata_path /work/ecg_info.tsv --encoder_path /app/encoder_ecg_4096_std.keras --encoder_layer activation_13 --longitudinal_path /app/transformer_ecg_only_v2026_03_04.keras --output /work/embeddings.csv –max_seq_len 256 --ecg_input_shape 4096
```
For longitudinal age inference:
```bash
docker run --rm -v "~/ecg_xml:/work" multimodal_finngen_deploy --mode infer --path /work/xmls --metadata_path /work/ecg_info.tsv --encoder_path /app/encoder_ecg_5000_std.keras --encoder_layer dropout_2 --longitudinal_path /app/ecgs2age_v2026_03_31.keras --output /work/embeddings_w_delta_age.csv --max_seq_len 512 --delta_consider --ecg_input_shape 5000
```
If it works, you should see the output in `~/ecg_xml`. Then save your docker image as tarball:
```bash
docker save multimodal_finngen_deploy:latest -o multimodal_finngen_deploy.tar
```

## Deploy to FinnGEN
Download the tarball (maybe a huge 20GB+ file). Then split it into smaller files, because FinnGEN has a limit of 5GB per file:
```bash
split -b 2300M multimodal_finngen_deploy.tar multimodal_finngen_deploy_part_
```
Login to your finngen account and navigate to the green bucket Google Console page. 
The address depends on the sandbox version. Currently, it is at: [https://console.cloud.google.com/storage/browser/fg-production-sandbox-54_greenuploads/<folder_name>](https://console.cloud.google.com/storage/browser/fg-production-sandbox-54_greenuploads/<folder_name>).
Upload all the parts here. Then after they pass the virus scan, which takes ~20 minutes, they will show up in your FinnGEN IVM at the path `/finngen/green/<folder_name>`.
You can replace `folder_name` with any folder name you want, but must be consistent between the upload and the IVM path.


More docs are here: [https://docs.finngen.fi/working-in-the-sandbox/quirks-and-features/how-to-upload-to-your-own-ivm-via-finngen-green](https://docs.finngen.fi/working-in-the-sandbox/quirks-and-features/how-to-upload-to-your-own-ivm-via-finngen-green)

Once all the pieces have been uploaded, reassemble them in the sandbox:
```bash
cd /finngen/green/<folder_name>
cat multimodal_finngen_deploy_part_* > ~/multimodal_finngen_deploy.tar
```

Load the docker image:
```bash
cd ~
docker load -i multimodal_finngen_deploy.tar
```
Then run the docker image:  
Example command for multimodal inference:
```
docker run -v /finngen/library-red/EA3_HEART_FAILURE_1.0/data:/work -v /home/ivm/output:/output multimodal_finngen_deploy --mode infer --path /work/ecg --metadata_path /work/EA3_HEART_FAILURE_ecg_info_1.0.txt --encoder_path /app/encoder_ecg_4096_std.keras --encoder_layer activation_13 --longitudinal_path /app/transformer_ecg_only_v2026_03_04.keras --output /output/embeddings.csv –max_seq_len 256 --ecg_input_shape 4096
```
