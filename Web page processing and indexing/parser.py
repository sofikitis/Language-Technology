import os
from bs4 import BeautifulSoup

read_path = 'html_pages/'
write_path = 'txt_files/'

entries = os.listdir(read_path)

for i in entries:
    r_sub_path = read_path + i

    with open(r_sub_path, 'rb') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'lxml')
    f.close()

    txt_file = soup.get_text()

    txt_name = write_path + i
    with open(txt_name, 'w') as f:
        f.write(txt_file)
    f.close()
