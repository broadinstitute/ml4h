# Make a deployment docker with a model from the Model Factory
Edit `Dockerfile` to copy and load your `.keras` model file.
Then build the docker image:
```bash 
docker build -t ecg_supervised_finngen_deploy .
```
Then run the docker image:
```bash 
docker run --rm -v "~/ecg_xml:/work" ecg_supervised_finngen_deploy --directory /work/xmls --model_path /app/ecg_cnn_scratch_26task_v2025_08_11.keras --output_file /work/supervised.csv --metadata /work/ecg_info.tsv --ecg_input_shape 4096
```
If it works, you should see the output in `~/ecg_xml`. Then save your docker image as tarball:
```bash
docker save ecg_supervised_finngen_deploy:latest -o ecg_supervised_finngen_deploy.tar
```

## Deploy to FinnGEN
Download the tarball (maybe a huge 20GB+ file). Then split it into smaller files, because FinnGEN has a limit of 5GB per file:
```bash
split -b 2300M ecg_supervised_finngen_deploy.tar ecg_supervised_finngen_deploy_part_
```
Login to your finngen account and navigate to the green bucket Google Console page. 

The address depends on the sandbox version. Currently, it is at: [https://console.cloud.google.com/storage/browser/fg-production-sandbox-54_greenuploads/<folder_name>](https://console.cloud.google.com/storage/browser/fg-production-sandbox-54_greenuploads/<folder_name>).
Upload all the parts here. Then after they pass the virus scan, which takes ~20 minutes, they will show up in your FinnGEN IVM at the path `/finngen/green/<folder_name>`.
You can replace `<folder_name>` with any folder name you want, but must be consistent between the upload and the IVM path.



More docs are here: [https://docs.finngen.fi/working-in-the-sandbox/quirks-and-features/how-to-upload-to-your-own-ivm-via-finngen-green](https://docs.finngen.fi/working-in-the-sandbox/quirks-and-features/how-to-upload-to-your-own-ivm-via-finngen-green)

Once all the pieces have been uploaded, reassemble them in the sandbox:
```bash
cd /finngen/green/<folder_name>
cat ecg_supervised_finngen_deploy_part_* > ~/ecg_supervised_finngen_deploy.tar
```

Load the docker image:
```bash
cd ~
docker load -i ecg_supervised_finngen_deploy.tar
```
Then run the docker image:
```
docker run -v /finngen/library-red/EA3_HEART_FAILURE_1.0/data:/work -v /home/ivm/output:/output ecg_supervised_finngen_deploy --directory /work/ecg --metadata /work/EA3_HEART_FAILURE_ecg_info_1.0.txt --model_path /app/ecg_cnn_scratch_26task_v2025_08_11.keras --output_file /output/embeddings_supervised.csv --ecg_input_shape 4096
```
