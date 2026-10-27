"""Local browser-check API. Disposable in-memory records; ML outputs are fixtures.

Run only for UI checks: python -m checks.browser_service
Then point Streamlit's API_URL at http://127.0.0.1:18001.
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
from checks.test_accounts import AccountTests


class Handler(BaseHTTPRequestHandler):
    def handle_request(self):
        data = self.rfile.read(int(self.headers.get('Content-Length', '0')))
        response = AccountTests.client.request(self.command, self.path, content=data,
                                               headers=dict(self.headers))
        self.send_response(response.status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(response.content)

    do_GET = handle_request
    do_POST = handle_request
    do_DELETE = handle_request

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    AccountTests.setUpClass()
    AccountTests('test_auth_required_expiry_and_logout').setUp()
    print('Disposable UI-check API: http://127.0.0.1:18001 (fixture predictions)', flush=True)
    HTTPServer(('127.0.0.1', 18001), Handler).serve_forever()
