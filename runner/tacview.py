import socket
import time

class Tacview(object):
    def __init__(self):
        self.host = "192.168.3.51"
        self.port = 41235
        self.setup_server()
        
    def setup_server(self):
        """初始化服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            print(f"Server listening on {self.host}:{self.port}")
            self.connect()
        except Exception as e:
            print(f"Setup error: {e}")
            self.cleanup()
            raise

    def send_data_to_client(self, data):
        """发送数据到客户端"""
        try:
            self.client_socket.send(data.encode())
        except Exception as e:
            print(f"Send error: {e}")
            self.reconnect()
            
    def connect(self):
        """建立连接"""
        try:
            print("Waiting for connection...")
            self.client_socket, self.address = self.server_socket.accept()
            print(f"Accepted connection from {self.address}")
            
            # 发送握手数据
            handshake_data = "XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\nHostUsername\n\x00"
            self.client_socket.send(handshake_data.encode())
            
            # 接收客户端响应
            data = self.client_socket.recv(1024)
            print(f"Received data from {self.address}: {data.decode()}")
            
            # 发送头部数据
            header_data = ("FileType=text/acmi/tacview\nFileVersion=2.1\n"
                          "0,ReferenceTime=2020-04-01T00:00:00Z\n#0.00\n")
            self.client_socket.send(header_data.encode())
            print("Connection established")
            
        except Exception as e:
            print(f"Connection error: {e}")
            self.cleanup()
            raise

    def reconnect(self):
        """重新连接"""
        print("Attempting to reconnect...")
        self.cleanup()
        self.setup_server()

    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, 'client_socket') and self.client_socket:
                self.client_socket.close()
                self.client_socket = None
            if hasattr(self, 'server_socket') and self.server_socket:
                self.server_socket.close()
                self.server_socket = None
        except Exception as e:
            print(f"Cleanup error: {e}")

    def __del__(self):
        """析构函数"""
        self.cleanup()