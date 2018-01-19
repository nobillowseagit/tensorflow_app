import random
import os

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
path_dir = "data/"
# 不存在直接创建
if not os.path.exists(path_dir):
    os.makedirs(path_dir)

# 随机的返回一个字符，这里不只限于生成一个字符（后期扩展性）
def random_number_text(char_set=number, code_size=1):
    code_text = []
    for each in range(code_size):
        c = random.choice(char_set)
        code_text.append(c)
    return code_text


def write_labels(size, name):
    code_list = []
    for each in range(size):
        number_list = random_number_text()
        code = ''.join(number_list)  # 用引号中的东西去连接list的两个条目
        code_list.append(code)
    code_text = '#'.join(code_list)
    print(code_text)
    f = open(path_dir + name, 'w')
    f.write(code_text)
    f.close()


def main():
    # 可指定大小
    trian_size = 5000
    test_size = 1000
    train_label_name = "code_train_text.txt"
    test_label_name = "code_test_text.txt"
    write_labels(trian_size, train_label_name)
    write_labels(test_size, test_label_name)


if __name__ == '__main__':
    main()
