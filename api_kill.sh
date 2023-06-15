#!/usr/bin/env bash
ps -ef|grep '/usr/bin/python3.10 api.py'|awk '{print $2}'| xargs kill -9