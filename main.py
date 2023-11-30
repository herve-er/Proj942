import API_Handler.API_handler
import server
import threading
ip = '192.168.39.37'
if __name__ == '__main__':

    web_server_thread = threading.Thread(target=server.start, args=(ip,))
    web_server_thread.start()
    API_Handler.API_handler.start(ip)
