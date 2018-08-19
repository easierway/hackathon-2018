nohup python wukong_service.py -s 672 -p 50672 -w  /home/ec2-user/src/wukong/tmp/douyin_672.combined_model_weightsacc0.921_val_acc0.993.best.hdf5 >> 50672.log 2>&1 &
nohup python wukong_service.py -s 672 -p 51672 -w  /home/ec2-user/src/wukong/tmp/douyin_672.combined_model_weightsacc0.938_val_acc0.948.best.hdf5 >> 51672.log 2>&1 &

