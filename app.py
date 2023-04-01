import traceback

from flask import Flask, render_template
from flask_cors import CORS
from flask_executor import Executor
from flask_socketio import SocketIO, emit
from gevent import monkey

from main import run

monkey.patch_all(ssl=False)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent', logger=True)
cors = CORS(app)
executor = Executor(app)

executor.init_app(app)
app.config['EXECUTOR_MAX_WORKERS'] = 5

@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('message')
def handle_message(data):
    question = data['question']
    print("question: " + question)

    if executor.futures:
        emit('response', {'response': 'Server is busy, please try again later'})
        return

    try:
        future = executor.submit(run, question)
        response = future.result()
        emit('response', {'response': response})
    except Exception as e:
        traceback.print_exc()
        # print(f"Error processing request: {str(e)}")
        emit('response', {'response': 'Server is busy. Please try again later.'})


if __name__ == '__main__':
    socketio.run(app, use_reloader=True)
