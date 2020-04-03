echo "Downloading project data..."
	
wget "https://www.dropbox.com/s/khda7iizffcaflt/jobs6.tsv?dl=1" 
mv "jobs6.tsv?dl=1" data/raw/jobs6.tsv
	
echo "Downloading Universal Sentence Encoder. This will take a bit."
curl -L "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed" | tar -zxvC src/models/use
