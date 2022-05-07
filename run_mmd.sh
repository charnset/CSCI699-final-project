BASE="squad"
BASE_PATH="in_domain_dev/SQuAD.jsonl.gz"
echo BASE: $BASE, BASE_PATH: $BASE_PATH
IN_DOMAIN=$1
OUT_DOMAIN=$2

echo IN_DOMAIN
for filename in $IN_DOMAIN/*.gz; do
	if [ $filename == $BASE_PATH ]
	then
		continue
	fi
	# echo $filename
	python playground.py -x $BASE_PATH -y $filename
done

echo OUT_DOMAIN
for filename in $OUT_DOMAIN/*.gz; do
	# echo $filename
	python playground.py -x $BASE_PATH -y $filename
done