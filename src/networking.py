import socket

def is_port_open(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        s.connect((ip, port))
        return True
    except socket.error:
        return False
    finally:
        s.close()

def find_open_ports(ip, port_range):
    open_ports = []
    for port in range(port_range[0], port_range[1] + 1):
        if is_port_open(ip, port):
            open_ports.append(port)
    return open_ports
