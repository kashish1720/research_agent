import ipaddress
import socket
import sys

hosts = sys.argv[1:] or ["www.firecrawl.dev", "redis.io", "www.zenml.io"]
for h in hosts:
    try:
        infos = socket.getaddrinfo(h, 443, proto=socket.IPPROTO_TCP)
    except Exception as e:
        print(h, "DNS ERROR", e)
        continue
    for ip in sorted({i[4][0] for i in infos}):
        print(h, ip, "is_global =", ipaddress.ip_address(ip).is_global)
