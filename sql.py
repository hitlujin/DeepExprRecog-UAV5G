#! /usr/bin/python
# -*- coding: UTF-8 -*-
import pymysql
from timeit import default_timer



def get_connection():
    conn = pymysql.connect(host=host, port=port, db=db, user=user, password=password)
    return conn


# ---- 使用 with 的方式来优化代码
class UsingMysql(object):

    def __init__(self, commit=True, log_time=True, log_label='总用时'):
        """

        :param commit: 是否在最后提交事务(设置为False的时候方便单元测试)
        :param log_time:  是否打印程序运行总时间
        :param log_label:  自定义log的文字
        """
        self._log_time = log_time
        self._commit = commit
        self._log_label = log_label

    def __enter__(self):

        # 如果需要记录时间
        if self._log_time is True:
            self._start = default_timer()

        # 在进入的时候自动获取连接和cursor
        conn = get_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        conn.autocommit = False

        self._conn = conn
        self._cursor = cursor
        return self

    def __exit__(self, *exc_info):
        # 提交事务
        if self._commit:
            self._conn.commit()
        # 在退出的时候自动关闭连接和cursor
        self._cursor.close()
        self._conn.close()

        if self._log_time is True:
            diff = default_timer() - self._start
            print('-- %s: %.6f 秒' % (self._log_label, diff))

    @property
    def cursor(self):
        return self._cursor

def check_it():
    with UsingMysql(log_time=True) as um:
        um.cursor.execute("select count(id) as total from t_face_expression_recognition")
        data = um.cursor.fetchone()
        print("-- 当前数量: %d " % data['total'])

def inser_face_filename(filename,expression_code,match_rate,detection_rate,recognition_rate,video_id=1,):
    with UsingMysql(log_time=True) as um:
        sql = "insert into t_face_expression_recognition(file_name, expression_code,match_rate,face_detection_rate,face_recognition_rate,video_id) values(%s, %s, %s, %s, %s, %s)"
        params = (filename, expression_code, match_rate, detection_rate, recognition_rate, video_id)
        print(params)
        um.cursor.execute(sql, params)


if __name__ == '__main__':
    check_it()
    inser_face_filename("3.jpg",1,0.99,0.99,0.99,1)