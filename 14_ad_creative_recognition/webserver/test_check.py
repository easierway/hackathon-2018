import subprocess

for i in range(3):
    retcode = subprocess.call(["python", 'wukong_check.py', '-p', '/home/ec2-user/src/leon-hackathon-2018/14_ad_creative_recognition/webserver/upload_image/2035.jpg'])
    print retcode

