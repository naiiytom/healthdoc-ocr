import os
import logging
from logging import Formatter, FileHandler
from flask import Flask, request, jsonify
from ocr import process_image

app = Flask(__name__)


@app.route(f'/', methods=['GET'])
def main():
    return jsonify({
        'status': 200,
        'text': 'Hello welcome to OCR backend'
    })


@app.route(f'/api/ocr', methods=['POST'])
def ocr():
    try:
        url = request.json['img']
        if ('jpg' in url or 'JPG' in url or 'png' in url):
            output = process_image(url)
            return jsonify({
                'status': 200,
                'output': output
            })
        else:
            return jsonify({
                'status': 400,
                'error': 'File should be image'
            })
    except:
        return jsonify({
            'status': 403,
            'error': 'Forbidden'
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    print('Running API server on: http://0.0.0.0:5000/')
