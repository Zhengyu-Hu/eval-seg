mkdir -p ./data

cd ./data
wget -O llff.zip \
https://www.dropbox.com/scl/fo/3rrlio2ht7evu1k5fxea4/ADmN3Rfwblug4fkIVBGys-o?rlkey=p24xpye3t329jmg2d5skwkx1u&st=9gax79mk&dl=1
wait
unzip llff.zip
cd ../