import unittest

from more_itertools import unzip

from datasets.Jiang.DatasetFilter import DatasetFilter
from datasets.Jiang.GitDiffOutputProcessor import GitDiffOutputProcessor


class DatasetFilterTest(unittest.TestCase):
    def test_binary_file_deletion(self):
        diff = 'deleted file mode 100644 <nl> index 22c5544 . . 0000000 <nl> Binary files a / app / src / main / res / mipmap - xxxhdpi / ic_launcher . png and / dev / null differ <nl>'
        self.assertFalse(DatasetFilter.validate(diff, ''))

    def test_multiple_changes(self):
        diff = 'deleted file mode 100644 <nl> index fadc72c . . 0000000 <nl> mmm a / GROUPS <nl> ppp / dev / null <nl> - streaming - compute <nl> deleted file mode 100644 <nl> index 5355fa4 . . 0000000 <nl> mmm a / OWNERS <nl> ppp / dev / null <nl> - jgrier <nl> - jwu <nl> - kramasamy <nl> - mfu <nl> - nlu <nl> - smittal <nl> - vikasr <nl>'
        self.assertFalse(DatasetFilter.validate(diff, ''))

    def test_file_content_modification(self):
        diff = 'mmm a / src / android / notification / Notification . java <nl> ppp b / src / android / notification / Notification . java <nl> public class Notification { <nl> * Notification type can be one of pending or scheduled . <nl> * / <nl> public Type getType ( ) { <nl> - return isTriggered ( ) ? Type . TRIGGERED : Type . SCHEDULED ; <nl> + return isScheduled ( ) ? Type . SCHEDULED : Type . TRIGGERED ; <nl> } <nl> / * * <nl>'
        self.assertTrue(DatasetFilter.validate(diff, ''))

    def test_file_content_modification_only_addition(self):
        diff = 'mmm a / CHANGELOG . md <nl> ppp b / CHANGELOG . md <nl> Changelog <nl> * Added * WRITE_CALENDAR * permissions to calendar restriction <nl> * Updated XposedBridge to version 54 ( Xposed version 2 . 6 is required now ) <nl> + * Show all usage data ( [ issue ] ( / . . / . . / issues / 1695 ) ) <nl> [ Open issues ] ( https : / / github . com / M66B / XPrivacy / issues ? state = open ) <nl>'
        self.assertTrue(DatasetFilter.validate(diff, ''))

    def test_rename_file(self):
        diff = 'similarity index 96 % <nl> rename from fml - src - 3 . 0 . 58 . 278 . zip <nl> rename to fml - src - 3 . 0 . 60 . 279 . zip <nl> Binary files a / fml - src - 3 . 0 . 58 . 278 . zip and b / fml - src - 3 . 0 . 60 . 279 . zip differ <nl>'
        self.assertFalse(DatasetFilter.validate(diff, ''))

    def test_file_addition(self):
        diff = 'new file mode 100644 <nl> index 0000000 . . 8b79786 <nl> Binary files / dev / null and b / third_party / truth / truth - 0 . 28 . jar differ <nl>'
        self.assertFalse(DatasetFilter.validate(diff, ''))


if __name__ == '__main__':
    unittest.main()
