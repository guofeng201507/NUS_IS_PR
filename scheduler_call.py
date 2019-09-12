import time, threading
import requests

def foo():
    print(time.ctime())
    r = requests.get("http://sharkv2.ap-southeast-1.elasticbeanstalk.com/crawl")
    print(r.text)
    threading.Timer(43200, foo).start()

foo()

#output:
#Tue Sep  3 20:19:47 2019
#Data has been crawled and analysed.