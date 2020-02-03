#%%
import socket

host=socket.gethostname()
print(host)
ip=socket.gethostbyname(host)
print(ip)
#%%
from wifi import Cell,Scheme
Cell.all('wlan0')

