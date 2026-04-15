import http.server
import socketserver
import webbrowser
import threading

PORT = 8000

# Serve files from current directory
Handler = http.server.SimpleHTTPRequestHandler

def open_browser():
    webbrowser.open(f"http://localhost:{PORT}/sift_ui.html")

if __name__ == "__main__":
    # Open browser after a short delay so server is ready
    threading.Timer(1.0, open_browser).start()

    with socketserver.TCPServer(("localhost", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        print("Opening browser to sift_ui.html...")
        httpd.serve_forever()