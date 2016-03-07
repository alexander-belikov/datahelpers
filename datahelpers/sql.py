from pymysql import connect
from pandas import read_sql, DataFrame
from json import load
from os.path import expanduser, join


session_dir = '.mysql'
session_file_prefix = 'session_'

q1 = """
SELECT
     table_schema as `Database`,
     table_name AS `Table`,
     round(((data_length + index_length) / 1024 / 1024), 2) `Size in MB`
FROM information_schema.TABLES
ORDER BY (data_length + index_length) DESC;
"""


def get_session_info(session_name):
    """
    arguments:
    :param session_name: the session suffix identifying the file ~/.mysql/
        from which the session dictionary is loaded
    """

    data = {}
    if session_name:
        fpath = join(expanduser('~'), session_dir,
                     session_file_prefix+session_name)
        with open(fpath, 'r') as f:
            data = load(f)
    return data


def get_info(session=None, hostname=None, username=None, password=None):
    """

    :param session: the session suffix identifying the file ~/.mysql/
        from which the session dictionary is loaded
    :param hostname: the hostname the mysql db is located
    :param username: the username to access the mysql db
    :param password: the password to access the mysql db
    :return: pandas DataFrame containing columns for
        database name, table name and size in Mb
    """
    session_dict = get_session_info(session)
    if hostname:
        session_dict['host'] = hostname
    if username:
        session_dict['user'] = username
    if password:
        session_dict['passwd'] = password
    conn = connect(**session_dict)
    cur = conn.cursor()
    cur.execute(q1)
    ll = []
    for row in cur.fetchall():
        ll.append(row)
    df = DataFrame(ll, columns=['db', 'table', 'size'])
    conn.close()
    return df


def get_table(session=None, hostname=None, username=None,
              password=None, database='valentin', table='GeneWAys',
              query_dict=None):
    """
    Parameters
    ----------
    :param session: string
        Session suffix identificator
    :param hostname: string
        Hostname of mysql db
    :param username: string
        Username to access mysql db
    :param password: string
        Password to access mysql db
    :param database: string
        Name of the database
    :param table: string
        Name of the table
    :param query_dict: dict
        dictionary of given structure
            qq = {'columns': ['pmid', 'issn', 'year'],
            'mask': {'pmid': [9988722, 3141384, 1924363]}, 'nrows':5}
        any key can be left out
    :return: cross-section of a table from a database
    """
    session_dict = get_session_info(session)
    if hostname:
        session_dict['host'] = hostname
    if username:
        session_dict['user'] = username
    if password:
        session_dict['passwd'] = password
    session_dict['db'] = database

    conn = connect(**session_dict)

    if query_dict and 'columns' in query_dict and query_dict['columns']:
        cols = ', '.join(map(lambda x: str(x), query_dict['columns']))
    else:
        cols = '*'
    query = 'select ' + cols + ' from ' + table
    if query_dict and 'mask' in query_dict and query_dict['mask']:
        cs = ' where '
        for k in query_dict['mask'].keys():
            cs += k + ' in (%s)'
            ids = ', '.join(map(lambda x: str(x), query_dict['mask'][k]))
            cs %= ids
        query += cs

    if query_dict['nrows']:
        query += ' limit ' + str(query_dict['nrows'])
    query += ';'
    df = read_sql(query, con=conn)
    conn.close()
    return df
