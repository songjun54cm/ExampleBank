__author__ = "JunSong<songjun@kuaishou.com>"

# Date: 2020/12/17
import sys
import io


def merge_map(source_map, target_map):
    if isinstance(source_map, dict):
        for k, v in source_map.items():
            if k in target_map:
                merge_map(v, target_map[k])
            else:
                target_map[k] = v


def form_robort_content(content_map, tab_size):
    content = []
    if isinstance(content_map, dict):
        for k, v in content_map.items():
            content.append("%s%s" % (" " * tab_size, k))
            content.extend(form_robort_content(v, tab_size + 4))
    else:
        content.append("%s%s" % (" " * tab_size, content_map))
    return content


class RobortContent():
    def __init__(self, in_file, title, head, reformate_fn, use_extend=False):
        self.title = title
        self.content_map = {}
        self.formate(in_file, title, head, reformate_fn, use_extend)

    def formate(self, in_file, title, head, reformate_fn, use_extend=False):
        print("read %s" % in_file)
        with io.open(in_file, "r", encoding='UTF-8') as in_f:
            line = in_f.readline()
            while line:
                line = line.strip().replace(r'\N', '')
                merge_map(reformate_fn(line.split("\t")), self.content_map, )
                line = in_f.readline()

    def to_robort_string(self):
        content = []
        content.append("<< %s >>" % self.title)
        # content.append("<font face=\\\"黑体\\\" size=5>%s</font>" % self.title)
        content.extend(form_robort_content(self.content_map, 0))
        return "\\n".join(content)


def form_campaign_chengben(in_file, title):
    def reformate(vals):
        cmap = {}
        campaign, ucnt, _,_,_, r1, r2, r3 = vals
        campaign_vals = "unit数:%s, 欠:%.4f, 达:%.4f, 超:%.4f" % \
                        (ucnt, float(r1), float(r2), float(r3))
        cmap[campaign] = campaign_vals
        return cmap
    robort_content = RobortContent(in_file, title, "", reformate)
    return robort_content


def form_action_chengben(in_file, title):
    def reformate(vals):
        campaign, action, ucnt, _,_,_, r1,r2,r3 = vals
        act_vals = "unit数:%s, 欠:%.4f, 达:%.4f, 超:%.4f" % \
                    (ucnt, float(r1), float(r2), float(r3))
        cmap = {
          campaign: {
            action: act_vals
          }
        }
        return cmap
    robort_content = RobortContent(in_file, title, "", reformate)
    return robort_content


def main(argv):
    date = "20201217"
    output_file = "test1_out.txt"
    campaign_chengben_file = "campaign_chengben_file.txt"
    action_chengben_file = "action_chengben_file.txt"
    content_list = []
    content_list.append(form_campaign_chengben(campaign_chengben_file, "计划维度大于5A的单元成本情况"))
    content_list.append(form_action_chengben(action_chengben_file, "action维度大于5A的单元成本情况"))

    str_list = []
    for c in content_list:
        str_list.append(c.to_robort_string())

    print("\\n====================\\n".join(str_list))




if __name__ == "__main__":
    main(sys.argv[1:])