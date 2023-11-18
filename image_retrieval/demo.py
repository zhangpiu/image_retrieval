#!/usr/bin/python3
# coding: utf-8
# author: zhangpiu

import argparse
import base64
from datetime import datetime
import glob
import imp
import io
import json
import logging
import os
import sys
import web
import random
import requests
import uuid

from PIL import Image
import torch

# Http server related
urls = ('/', 'HomeHandler', '/resource/(.*)', 'ResourceHandler')

page_template = """
<head>
  <meta charset="UTF-8">
</head>
<html>
<body>
<h1><b>Search %d results from %d ImageNet images! Time cost: %d(ms).</b></h1>
<form action="" method="POST" enctype="multipart/form-data">
  <input type="file" name="img">
  <input type="submit">
  <span style="padding-left:150px">
    <input type="button" value="I'm feeling lucky" onclick="location.href='/?q=%s';">
  </span>
</form>
%s
</body>
</html>
"""


def generate_neighbor_cell_html(neighbor):
  payload = json.loads(base64.b64decode(neighbor["payload"]).decode())
  image_resource = os.path.join('/resource', payload["img"])
  return ("<td width='180px' valign='top'>"
          "<img width='180px' src='%s'>"
          "<font size='2pt'>"
          "<br><b>ID</b>: %s"
          "<br><b>类目</b>: %s(%d)"
          "<br><b>距离</b>: %.3f"
          "</font>"
          "</td>") % (image_resource, neighbor["id"], "", neighbor["label"],
                      neighbor["distance"])


def generate_query_cell_html(image_data, display_name):
  if image_data is None:
    return ''
  encoded = base64.b64encode(image_data)
  content = ("<td width='400px' valign='top'>"
             "<img width='400px' src='data:image/png;base64, %s'>"
             "<font size='2pt'>"
             "<br><b>%s</b>"
             "</font>"
             "</td>") % (encoded.decode(), display_name)
  return content


def generate_input_html(image):
  image_cells = []
  if image is not None:
    image_cells.append(generate_query_cell_html(image, u'原图'))
  content = "<h2>Input</h2><table border=1><tr>%s</tr></table>" % (
      ''.join(image_cells))
  return content


class ResourceHandler:

  def GET(self, name):
    web.header('Content-Type', 'image/jpeg', unique=True)
    filepath = os.path.join(args.resource, name)
    if os.path.exists(filepath):
      with open(filepath, 'rb') as file:
        content = file.read()
        return content
    else:
      raise web.notfound()


def handler(img_data: bytes, query: str, save=False):
  try:
    img = Image.open(io.BytesIO(img_data))
    if save:
      now = datetime.now()
      filename = os.path.join(
          args.dir, '{}_{}.jpg'.format(now.strftime("%Y_%m_%d_%H_%M_%S"),
                                       uuid.uuid4()))
      if img.mode != 'RGB':
        img = img.convert('RGB')
      img.save(filename)

    tensor = transform(img)
    feature = model(torch.stack([tensor]))[0]
  except Exception as e:
    logging.error("Invalid input image: {}".format(str(e)))
    return page_template % (0, 0, 0, query, "<hr><b>Invalid image file!</b>")

  body = dict(query=feature.tolist(), top_k=50)
  response = requests.post('http://{}/search'.format(args.target),
                           data=json.dumps(body))

  if response.status_code != 200:
    return page_template % (0, 0, 0, "",
                            "<hr><b>Error occurred when calling ann!</b>")
  result = json.loads(response.content)
  content = [
      generate_input_html(img_data), "<br><h2>Output</h2><table border=0>"
  ]
  for (i, neighbor) in enumerate(result['neighbors']):
    if i % 10 == 0:
      content.append("<tr>")
    content.append(generate_neighbor_cell_html(neighbor))
    if (i % 10 == 9) or (i + 1 == len(result['neighbors'])):
      content.append("</tr>")
  content.append("</table>")

  return page_template % (len(
      result['neighbors']), result['total_count'], result['search_cost_ms'],
                          query, "<hr>%s" % "".join(content))


class HomeHandler:

  def GET(self):
    web.header('Content-Type', 'text/html; charset=utf-8', unique=True)
    params = web.input()
    query = params.get("q", "")
    next_random_query = random.choice(list(images.keys()))
    if not query:
      return page_template % (0, 0, 0, next_random_query, "")

    if not os.path.exists(images[query]):
      return page_template % (0, 0, 0, next_random_query,
                              "<hr>%s not found!" % query)
    with open(images[query], 'rb') as file:
      return handler(file.read(), next_random_query, False)

  def POST(self):
    web.header('Content-Type', 'text/html; charset=utf-8', unique=True)
    data = web.input()
    if not data:
      return page_template % (0, 0, 0, "", "")
    next_random_query = random.choice(list(images.keys()))
    return handler(data['img'], next_random_query, True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-j',
                      '--job',
                      default='feature_extraction/resnet50.py',
                      help='The model used to predict.')
  parser.add_argument('-t',
                      '--target',
                      default='localhost:7000',
                      help='The host of ann search engine.')
  parser.add_argument('-d',
                      '--dir',
                      default='saved_images',
                      help='Directory to save query images.')
  parser.add_argument('-r',
                      '--resource',
                      default='.',
                      help='Resource directory.')

  args, unknown = parser.parse_known_args()
  sys.argv = [""] + unknown

  if not os.path.exists(args.resource):
    raise ValueError("No such directory: {}".format(args.resource))
  images = glob.glob(
      os.path.join(args.resource, '**', '*.JPEG'), recursive=True) + glob.glob(
          os.path.join(args.resource, '**', '*.jpg'), recursive=True)
  images = dict([
      (os.path.splitext(os.path.basename(path))[0], path) for path in images
  ])

  if not os.path.exists(args.dir):
    os.makedirs(args.dir)

  print("Loading model from job file {}".format(args.job))
  job = imp.load_source('job', args.job)
  model = job.model
  transform = job.transform

  app = web.application(urls, globals(), autoreload=False)
  app.run()
