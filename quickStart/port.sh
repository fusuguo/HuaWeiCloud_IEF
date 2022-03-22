#!/bin/bash
	ls -1rt /app/pic    | grep -E ".jpg|.png" > /app/pic/picture.txt
	ls -1rt /app/pic    | grep -E ".jpg|.png" | awk -F "." '{print $1 }'> /app/result.txt
	row=`wc -l < /app/pic/picture.txt`
for ((i =1;i <= $row;i=i+1))
do
	file=`sed -n "$i"p < /app/pic/picture.txt`
	name=`sed -n "$i"p < /app/result.txt`
if [ ! -f "/app/result/$name.json" ];then
	result=`curl -F 'images=@/app/pic/'$file'' -X POST http://$1:$2 -k`
	echo -n $result>>/app/result/$name.json
fi
done
