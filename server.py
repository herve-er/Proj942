from http.server import HTTPServer, SimpleHTTPRequestHandler, test
import sys
import os

# Port sur lequel le serveur écoutera (par défaut, 8000)
port = 8401

class CORSRequestHandler (SimpleHTTPRequestHandler):
    def end_headers (self):
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPRequestHandler.end_headers(self)



def start(ip):
    # init the server with root at "./website_config/website/"
    test(CORSRequestHandler, HTTPServer, port=port, bind=ip)


