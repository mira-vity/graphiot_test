import socket
import dpkt
import pandas as pd
import os
from natsort import natsorted
current_path = os.path.abspath(__file__)#当前目录
columns = ["direction","size","src_ip","dst_ip","src_mac","dst_mac","timestamp","time_interv"]#timestamp用于计算与下一个数据包的相差时间
GatewayMac = '14:cc:20:51:33:ea'

def getpcap(path):
    pcap = []
    for dirpah,dirnames,filenames in (os.walk(path)):#返回元组，以元组形式进行迭代
        for filename in filenames:
            pcap.append(filename)
    #print(pcap)
    return natsorted(pcap)

def pcap2csv(pacplist,path,csv_path):#
    for filename in pacplist:
        base_row = {c:[] for c in columns}#字典推导式
        fullpath = path+str(filename)
        #print (fullpath)
        timestamp_now = float(0)
        with open(fullpath,'rb')as f:
            pcapng = dpkt.pcapng.Reader(f)
            for timestamp,buf in pcapng:#以元组为对象进行迭代，注意迭代器只能遍历一次，不支持索引直接访问特定元素
                eth = dpkt.ethernet.Ethernet(buf)
                frame_size = len(eth)

                src_mac = ':'.join(f'{byte:02x}' for byte in eth.src)#每个字节转化为2个16进制数，以冒号分隔
                dst_mac = ':'.join(f'{byte:02x}' for byte in eth.dst)
                #print (src_mac+' '+GatewayMac)
                if (src_mac == GatewayMac):
                    direction = '-'
                    '''    
                    else: 
                        direction = '+'
                    '''
                elif (dst_mac == GatewayMac):  
                    direction = '+'
                else: continue#滤除除了网关和目标mac意外的其他mac

                if (eth.type == dpkt.ethernet.ETH_TYPE_IP):
                    ip = eth.data
                    #ip = dpkt.ip.IP(eth.data)
                    src_ip = None#socket.inet_ntoa(ip.src)
                    dst_ip = None#socket.inet_ntoa(ip.dst)
                elif (eth.type == dpkt.ethernet.ETH_TYPE_ARP):
                    ip = eth.data
                    src_ip = None
                    dst_ip = None
                
                
                new_row = {
                    "direction":direction,
                    "size":frame_size,
                    "src_ip":src_ip,
                    "dst_ip":dst_ip,
                    "src_mac":src_mac,
                    "dst_mac":dst_mac,
                    "timestamp":timestamp,
                    "time_interv":(0.01 if (timestamp-timestamp_now)>1 else 1),#当前数据包与上一个数据包的时间差值
                }
                #print (new_row)

                for c in base_row.keys():#取键
                    base_row[c].append(new_row[c])

                timestamp_now=timestamp#更新当前时间戳


        #关闭文件
        processed_df = pd.DataFrame(base_row)
        processed_df = processed_df.reset_index(drop=True)
        #processed_df.to_csv('./csv/'+str(filename)+".csv", index=False)
        processed_df.to_csv(csv_path+str(filename)+".csv", index=False)#按日期存储

if __name__ == "__main__":
    pass