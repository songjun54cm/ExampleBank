from datetime import datetime
from datetime import timedelta
import re

def get_date_list(seg):
    if '~' not in seg:
        return [seg]
    else:
        date_list = []
        start_p, end_p = seg.split('~')
        begin_date = datetime.strptime(start_p, "%Y%m%d")
        end_date = datetime.strptime(end_p, "%Y%m%d")
        cur_date = begin_date
        while cur_date <= end_date:
            date_list.append(cur_date.strftime("%Y%m%d"))
            cur_date = cur_date + timedelta(days=1)
        return date_list


def get_path_list(paths_str):
    if '~' not in paths_str:
        return [paths_str]
    segs = re.split(r"([/])", paths_str)
    prefix_list = [""]
    for seg in segs:
        date_list = get_date_list(seg)
        new_prefix_list = []
        for prefix in prefix_list:
            for s in date_list:
                new_prefix_list.append("%s%s" % (prefix, s))
        prefix_list = new_prefix_list
    return prefix_list


def main():
    a = "dir/songjun/20200525~20200604/*"
    a_list = get_path_list(a)
    print(a_list)
    b = "/home/songjun/20200525~20200604/*"
    b_list = get_path_list(b)
    print(b_list)


if __name__ == "__main__":
    main()

    a = "hello,world.happy/new,year."
    sentences = re.split(r"[.。!！?？；;，,\s+/]", a)
    print(sentences)

    a = "hello,world.happy/new,year."
    sentences = re.split(r"([.。!！?？；;，,\s+/])", a)
    print(sentences)
