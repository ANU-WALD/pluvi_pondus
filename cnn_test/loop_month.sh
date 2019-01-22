DATE=$(date -d "2017-01-01" -u)
END=$(date -d "2018-04-01" -u)
while [ "$DATE" != "$END" ]; do
	echo $DATE
	DATE=$(date -d "$DATE + 1 month" -u)
done
