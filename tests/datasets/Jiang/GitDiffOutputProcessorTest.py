import unittest

from more_itertools import unzip

from datasets.Jiang.GitDiffOutputProcessor import GitDiffOutputProcessor


class GitDiffOutputProcessorTest(unittest.TestCase):
    def test_binary_file_deletion(self):
        diff = 'deleted file mode 100644 <nl> index 22c5544 . . 0000000 <nl> Binary files a / app / src / main / res / mipmap - xxxhdpi / ic_launcher . png and / dev / null differ <nl>'
        prev_expected = 'Binary files a / app / src / main / res / mipmap - xxxhdpi / ic_launcher . png and / dev / null differ <nl>'
        updated_expected = 'deleted file mode 100644 <nl> Binary files a / app / src / main / res / mipmap - xxxhdpi / ic_launcher . png and / dev / null differ <nl>'
        prev_actual, updated_actual = GitDiffOutputProcessor.get_prev_and_updated(diff)
        self.assertEqual(prev_expected, prev_actual)
        self.assertEqual(updated_expected, updated_actual)

    def test_multiple_changes(self):
        # TODO: handle this situation
        diff = 'deleted file mode 100644 <nl> index fadc72c . . 0000000 <nl> mmm a / GROUPS <nl> ppp / dev / null <nl> - streaming - compute <nl> deleted file mode 100644 <nl> index 5355fa4 . . 0000000 <nl> mmm a / OWNERS <nl> ppp / dev / null <nl> - jgrier <nl> - jwu <nl> - kramasamy <nl> - mfu <nl> - nlu <nl> - smittal <nl> - vikasr <nl>'
        prev_expected = 'mmm a / GROUPS <nl> streaming - compute <nl> mmm a / OWNERS <nl> jgrier <nl> jwu <nl> kramasamy <nl> mfu <nl> nlu <nl> smittal <nl> vikasr <nl>'
        updated_expected = 'deleted file mode 100644 <nl> ppp / dev / null <nl> deleted file mode 100644 <nl> ppp / dev / null <nl>'
        prev_actual, updated_actual = GitDiffOutputProcessor.get_prev_and_updated(diff)
        self.assertEqual(prev_expected, prev_actual)
        self.assertEqual(updated_expected, updated_actual)

    def test_file_content_modification(self):
        diff = 'mmm a / src / android / notification / Notification . java <nl> ppp b / src / android / notification / Notification . java <nl> public class Notification { <nl> * Notification type can be one of pending or scheduled . <nl> * / <nl> public Type getType ( ) { <nl> - return isTriggered ( ) ? Type . TRIGGERED : Type . SCHEDULED ; <nl> + return isScheduled ( ) ? Type . SCHEDULED : Type . TRIGGERED ; <nl> } <nl> / * * <nl>'
        prev_expected = 'mmm a / src / android / notification / Notification . java <nl> public class Notification { <nl> * Notification type can be one of pending or scheduled . <nl> * / <nl> public Type getType ( ) { <nl> return isTriggered ( ) ? Type . TRIGGERED : Type . SCHEDULED ; <nl> } <nl> / * * <nl>'
        updated_expected = 'ppp b / src / android / notification / Notification . java <nl> public class Notification { <nl> * Notification type can be one of pending or scheduled . <nl> * / <nl> public Type getType ( ) { <nl> return isScheduled ( ) ? Type . SCHEDULED : Type . TRIGGERED ; <nl> } <nl> / * * <nl>'
        prev_actual, updated_actual = GitDiffOutputProcessor.get_prev_and_updated(diff)
        self.assertEqual(prev_expected, prev_actual)
        self.assertEqual(updated_expected, updated_actual)

    def test_file_content_modification_only_addition(self):
        diff = 'mmm a / CHANGELOG . md <nl> ppp b / CHANGELOG . md <nl> Changelog <nl> * Added * WRITE_CALENDAR * permissions to calendar restriction <nl> * Updated XposedBridge to version 54 ( Xposed version 2 . 6 is required now ) <nl> + * Show all usage data ( [ issue ] ( / . . / . . / issues / 1695 ) ) <nl> [ Open issues ] ( https : / / github . com / M66B / XPrivacy / issues ? state = open ) <nl>'
        prev_expected = 'mmm a / CHANGELOG . md <nl> Changelog <nl> * Added * WRITE_CALENDAR * permissions to calendar restriction <nl> * Updated XposedBridge to version 54 ( Xposed version 2 . 6 is required now ) <nl> [ Open issues ] ( https : / / github . com / M66B / XPrivacy / issues ? state = open ) <nl>'
        updated_expected = 'ppp b / CHANGELOG . md <nl> Changelog <nl> * Added * WRITE_CALENDAR * permissions to calendar restriction <nl> * Updated XposedBridge to version 54 ( Xposed version 2 . 6 is required now ) <nl> * Show all usage data ( [ issue ] ( / . . / . . / issues / 1695 ) ) <nl> [ Open issues ] ( https : / / github . com / M66B / XPrivacy / issues ? state = open ) <nl>'
        prev_actual, updated_actual = GitDiffOutputProcessor.get_prev_and_updated(diff)
        self.assertEqual(prev_expected, prev_actual)
        self.assertEqual(updated_expected, updated_actual)

    def test_file_addition(self):
        diff = 'new file mode 100644 <nl> index 0000000 . . 8b79786 <nl> Binary files / dev / null and b / third_party / truth / truth - 0 . 28 . jar differ <nl>'
        prev_expected = 'new file mode 100644 <nl> Binary files / dev / null and b / third_party / truth / truth - 0 . 28 . jar differ <nl>'
        updated_expected = 'Binary files / dev / null and b / third_party / truth / truth - 0 . 28 . jar differ <nl>'
        prev_actual, updated_actual = GitDiffOutputProcessor.get_prev_and_updated(diff)
        self.assertEqual(prev_expected, prev_actual)
        self.assertEqual(updated_expected, updated_actual)

    def test_all_examples(self):
        diffs = [
            'deleted file mode 100644 <nl> index 22c5544 . . 0000000 <nl> Binary files a / app / src / main / res / mipmap - xxxhdpi / ic_launcher . png and / dev / null differ <nl>',
            'deleted file mode 100644 <nl> index fadc72c . . 0000000 <nl> mmm a / GROUPS <nl> ppp / dev / null <nl> - streaming - compute <nl> deleted file mode 100644 <nl> index 5355fa4 . . 0000000 <nl> mmm a / OWNERS <nl> ppp / dev / null <nl> - jgrier <nl> - jwu <nl> - kramasamy <nl> - mfu <nl> - nlu <nl> - smittal <nl> - vikasr <nl>',
            'mmm a / src / android / notification / Notification . java <nl> ppp b / src / android / notification / Notification . java <nl> public class Notification { <nl> * Notification type can be one of pending or scheduled . <nl> * / <nl> public Type getType ( ) { <nl> - return isTriggered ( ) ? Type . TRIGGERED : Type . SCHEDULED ; <nl> + return isScheduled ( ) ? Type . SCHEDULED : Type . TRIGGERED ; <nl> } <nl> / * * <nl>',
            'mmm a / CHANGELOG . md <nl> ppp b / CHANGELOG . md <nl> Changelog <nl> * Added * WRITE_CALENDAR * permissions to calendar restriction <nl> * Updated XposedBridge to version 54 ( Xposed version 2 . 6 is required now ) <nl> + * Show all usage data ( [ issue ] ( / . . / . . / issues / 1695 ) ) <nl> [ Open issues ] ( https : / / github . com / M66B / XPrivacy / issues ? state = open ) <nl>',
            'new file mode 100644 <nl> index 0000000 . . 8b79786 <nl> Binary files / dev / null and b / third_party / truth / truth - 0 . 28 . jar differ <nl>'
        ]
        prev_expected = [
            'Binary files a / app / src / main / res / mipmap - xxxhdpi / ic_launcher . png and / dev / null differ <nl>',
            'mmm a / GROUPS <nl> streaming - compute <nl> mmm a / OWNERS <nl> jgrier <nl> jwu <nl> kramasamy <nl> mfu <nl> nlu <nl> smittal <nl> vikasr <nl>',
            'mmm a / src / android / notification / Notification . java <nl> public class Notification { <nl> * Notification type can be one of pending or scheduled . <nl> * / <nl> public Type getType ( ) { <nl> return isTriggered ( ) ? Type . TRIGGERED : Type . SCHEDULED ; <nl> } <nl> / * * <nl>',
            'mmm a / CHANGELOG . md <nl> Changelog <nl> * Added * WRITE_CALENDAR * permissions to calendar restriction <nl> * Updated XposedBridge to version 54 ( Xposed version 2 . 6 is required now ) <nl> [ Open issues ] ( https : / / github . com / M66B / XPrivacy / issues ? state = open ) <nl>',
            'new file mode 100644 <nl> Binary files / dev / null and b / third_party / truth / truth - 0 . 28 . jar differ <nl>'
        ]
        updated_expected = [
            'deleted file mode 100644 <nl> Binary files a / app / src / main / res / mipmap - xxxhdpi / ic_launcher . png and / dev / null differ <nl>',
            'deleted file mode 100644 <nl> ppp / dev / null <nl> deleted file mode 100644 <nl> ppp / dev / null <nl>',
            'ppp b / src / android / notification / Notification . java <nl> public class Notification { <nl> * Notification type can be one of pending or scheduled . <nl> * / <nl> public Type getType ( ) { <nl> return isScheduled ( ) ? Type . SCHEDULED : Type . TRIGGERED ; <nl> } <nl> / * * <nl>',
            'ppp b / CHANGELOG . md <nl> Changelog <nl> * Added * WRITE_CALENDAR * permissions to calendar restriction <nl> * Updated XposedBridge to version 54 ( Xposed version 2 . 6 is required now ) <nl> * Show all usage data ( [ issue ] ( / . . / . . / issues / 1695 ) ) <nl> [ Open issues ] ( https : / / github . com / M66B / XPrivacy / issues ? state = open ) <nl>',
            'Binary files / dev / null and b / third_party / truth / truth - 0 . 28 . jar differ <nl>'
        ]
        output = GitDiffOutputProcessor.get_prev_and_updated_for_diffs(diffs)
        prev_actual, updated_actual = list(unzip(output))
        self.assertEqual(prev_expected, list(prev_actual))
        self.assertEqual(updated_expected, list(updated_actual))


if __name__ == '__main__':
    unittest.main()
