#!/usr/bin/env bash
ps -ef|grep '/usr/bin/python3.10 web.py'|awk '{print $2}'| xargs kill -9