install:
	python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt

gitpod:
	sudo apt-get update && sudo apt-get install libgl1

azure:
	git config --global http.postBuffer 157286400
	git gc --aggressive
	git remote remove azure
	git remote add azure https://demorestapimsds498.scm.azurewebsites.net:443/demorestapimsds498.git
	git push azure

gcloud:
	curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-417.0.1-linux-x86_64.tar.gz
	tar -xf google-cloud-cli-417.0.1-linux-x86_64.tar.gz 