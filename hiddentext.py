from PIL import Image
import optparse

def str2bin(message):
    message_bytes = bytes(message, 'ascii')
    return "".join(["{:08b}".format(x) for x in message_bytes])

def bin2str(binary):
    binary_split = []
    count = 0
    temp = ""
    for i in range(len(binary)):
        count += 1
        temp += binary[i]
        if count == 8:
            binary_split.append(temp)
            count = 0
            temp = ""
    return "".join([chr(int(b, 2)) for b in binary_split])
    
def encode(num, digit):
    '''
    1. change num to binary.
    2. add digit to last digit of num binary.
    3. change num binary back to decimal, and return it.
    '''
    bin_num = bin(num)
    bin_num = bin_num[:-1] + digit
    return int(bin_num, 2)

def decode(num):
    return bin(num)[-1]

def hide(filename, message):
    img = Image.open(filename)
    binary = str2bin(message) + "1111111111111110"
    if img.mode == 'RGBA':
        datas = img.getdata()
        newData = []
        count = 0
        message_end = False
        for data in datas:
            if not message_end:
                newpix = []
                for num in data :
                    if count < len(binary):
                        newpix.append(encode(num, binary[count]))
                        count += 1
                    else:
                        newpix.append(num)
                        message_end = True
                newData.append(tuple(newpix))
            else:
                break
        img.putdata(newData)
        img.save(filename[:-4]+"write"+filename[-4:], "PNG")
        return "Completed!"
    else:
        return "Incorrect Image Mode, couldn't hide :("

def retr(filename):
    img = Image.open(filename)
    binary = ""
    
    if img.mode == 'RGBA':
        datas = img.getdata()
        for data in datas:
            for num in data:
                binary += decode(num)
                if binary[-16:] == "1111111111111110":
                    print("Seccuss!")
                    return bin2str(binary[:-16])
        return bin2str(binary)
    return "Incorrect Image Mode, couldn't retrieve :("

def main():
    parser = optparse.OptionParser('python a.py ' + '-e/-d <target file>')
    parser.add_option('-e', dest = 'hide', type='string', help='target pic path to hide text')
    parser.add_option('-d', dest = 'retr', type='string', help='target pic path to retrieve text')
    (options, args) = parser.parse_args()
    if options.hide != None:
        text = input("Enter a message to hide: ")
        print(hide(options.hide, text))
    elif options.retr != None:
        print(retr(options.retr))
    else:
        print(parser.usage)
        quit()

if __name__ == '__main__':
    main()          