import API_Handler.API_handler
import server
import threading

if __name__ == '__main__':

    web_server_thread = threading.Thread(target=server.start)
    web_server_thread.start()
    API_Handler.API_handler.start()
