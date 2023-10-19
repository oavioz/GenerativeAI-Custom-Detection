import oracledb
import numpy as np
import json
connection = None


DSN = "(description= (retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1521)(host=adb.il-jerusalem-1.oraclecloud.com))(connect_data=(service_name=g94749f4930ffe0_war1_low.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))"


async def insertRecord(data):
    connection = oracledb.connect(
    user="ADMIN",
    password="Unbroken-Neurotic0-Afford",
    dsn=DSN, 
    )  # the connection string copied from the cloud console
    #if connection is None:
    # await connectToDB()
    print("Successfully connected to Oracle Database")
    user_id =-1;
    with connection.cursor() as cursor:
        new_id = cursor.var(oracledb.NUMBER)
        row = cursor.execute("INSERT INTO ADMIN.AIFILES (img_url,img_path) VALUES (:0,:1) returning id into :2",[data["img_url"],data["img_path"],new_id])
        user_id = new_id.getvalue()    
        print("user_id",user_id)
       
    connection.commit()
    return {"user_id":int(user_id[0])} 


async def getFileByURL(url):
    connection = oracledb.connect(
    user="ADMIN",
    password="Unbroken-Neurotic0-Afford",
    dsn=DSN,
    )  # the connection string copied from the cloud console
    #if connection is None:
    # await connectToDB()
    print("Successfully connected to Oracle Database")
   
    meta =[]
    encode =[]
    with connection.cursor() as cursor:
        for row in cursor.execute('SELECT id,img_url,img_path FROM ADMIN.AIFILES where img_url=:0',[url]):
            print(row)
            line = {"id":row[0],"img_url":row[1],"img_path":row[2]}
            return line           
    return None

async def getFileByPath(url):
    print('getFileByPath',url)
    connection = oracledb.connect(
    user="ADMIN",
    password="Unbroken-Neurotic0-Afford",
    dsn=DSN,
    )  # the connection string copied from the cloud console
    #if connection is None:
    # await connectToDB()
    print("Successfully connected to Oracle Database")
   
    meta =[]
    encode =[]
    with connection.cursor() as cursor:
        for row in cursor.execute('SELECT id,img_url,img_path FROM ADMIN.AIFILES where img_path=:0',[url]):
            print(row)
            line = {"id":row[0],"img_url":row[1],"img_path":row[2]}
            return line           
    return None