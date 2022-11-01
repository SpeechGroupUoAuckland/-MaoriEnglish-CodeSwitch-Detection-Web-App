from waitress import serve
from webapi import app

serve(app, host='127.0.0.1', port=8500, ident=None)
