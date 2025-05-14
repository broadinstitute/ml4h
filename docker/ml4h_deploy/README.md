# Make a deployment docker with a model from the Model Factory
Edit `Dockerfile` and `process_files.py` to copy and load your `.keras` model file.
Then build the docker image:
```bash 
docker build -t ecg2hf_finngen_deploy .
```
Then run the docker image:
```bash 
docker run -v /home/sam/ecg_xml:/data -v /home/sam:/output ecg2hf_finngen_deploy
```
If it works, you should see the output in `/home/sam`. Then save your docker image as tarball:
```bash
docker save ecg2hf_finngen_deploy:latest -o ecg2hf_finngen_deploy.tar
```

## Deploy to FinnGEN
Download the tarball (maybe a huge 20GB+ file). Then split it into smaller files:
```bash
split -b 2300M ecg2hf_finngen_deploy.tar ecg2hf_finngen_deploy_part_
```
Login to your finngen account and navigate to the green bucket Google Console page. 
The address depends on the sandbox version. Currently, it is at: [https://console.cloud.google.com/storage/browser/fg-production-sandbox-6_greenuploads/sam](https://console.cloud.google.com/storage/browser/fg-production-sandbox-6_greenuploads/sam).
Upload all the parts here and then they will show up in your FinnGEN IVM at the path `/finngen/green/sam`.
You can replace `sam` with any folder name you want, but must be consistent between the upload and the IVM path.


More docs are here: [https://docs.finngen.fi/working-in-the-sandbox/quirks-and-features/how-to-upload-to-your-own-ivm-via-finngen-green](https://docs.finngen.fi/working-in-the-sandbox/quirks-and-features/how-to-upload-to-your-own-ivm-via-finngen-green)

Once all the pieces have been uploaded, reassemble them in the sandbox:
```bash
cd /finngen/green/sam
cat ecg2hf_finngen_deploy_part_* > ~/ecg2hf_finngen_deploy.tar
```

Load the docker image:
```bash
cd ~
docker load -i ecg2hf_finngen_deploy.tar
```
Then run the docker image:
```