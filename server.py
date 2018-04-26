import socket
import Access as ac

serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind(('143.215.106.194',8089))
serversocket.listen(5)


while True:
    connection, address = serversocket.accept()
    buf = connection.recv(64)

    if buf == b'to_open':
        ans = ac.Access()
        if ans:
            connection.send(b'True')
        else:
            connection.send(b'False')

    if buf == b'stop':
        break


    
