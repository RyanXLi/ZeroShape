python evaluate.py --yaml=options/shape.a40.wdecay.yaml --name=fourcards-a40-wdecay --data.dataset_test=ocrtoc --eval.vox_res=128 --eval.brute_force --eval.batch_size=1 --resume | tee wdecay.ocrtoc.log
sleep 10
python evaluate.py --yaml=options/shape.a40.wdecay.yaml --name=fourcards-a40-wdecay --data.dataset_test=omniobj3d --eval.vox_res=128 --eval.brute_force --eval.batch_size=1 --resume | tee wdecay.omniobj3d.log
sleep 10
python evaluate.py --yaml=options/shape.a40.wdecay.yaml --name=fourcards-a40-wdecay --data.dataset_test=pix3d --eval.vox_res=128 --eval.brute_force --eval.batch_size=1 --resume | tee wdecay.pix3d.log