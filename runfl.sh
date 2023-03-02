algo=2

python server.py > server.log &
sleep 2
for client_id in `seq 0 9`
do
python main.py $client_id --algorithm $algo > clients.log &
done
wait