---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# How to extract substrings using regular expressions

+++

Let's consider the following log data

```{code-cell} ipython3
import awkward as ak

lines = ak.from_iter(
    [
        "12-17 19:31:36.263  1795  1825 I PowerManager_screenOn: DisplayPowerStatesetColorFadeLevel: level=1.0\r",
        "12-17 19:31:36.263  5224  5283 I SendBroadcastPermission: action:android.com.huawei.bone.NOTIFY_SPORT_DATA, mPermissionType:0\r",
        "12-17 19:31:36.264  1795  1825 D DisplayPowerController: Animating brightness: target=21, rate=40\r",
        "12-17 19:31:36.264  1795  1825 I PowerManager_screenOn: DisplayPowerController updatePowerState mPendingRequestLocked=policy=BRIGHT, useProximitySensor=true, useProximitySensorbyPhone=true, screenBrightness=33, screenAutoBrightnessAdjustment=0.0, brightnessSetByUser=true, useAutoBrightness=true, blockScreenOn=false, lowPowerMode=false, boostScreenBrightness=false, dozeScreenBrightness=-1, dozeScreenState=UNKNOWN, useTwilight=false, useSmartBacklight=true, brightnessWaitMode=false, brightnessWaitRet=true, screenAutoBrightness=-1, userId=0\r",
        "12-17 19:31:36.264  1795  2750 I PowerManager_screenOn: DisplayPowerState Updating screen state: state=ON, backlight=823\r",
        "12-17 19:31:36.264  1795  2750 I HwLightsService: back light level before map = 823\r",
        "12-17 19:31:36.264  1795  1825 D DisplayPowerController: Animating brightness: target=21, rate=40\r",
        "12-17 19:31:36.264  1795  1825 V KeyguardServiceDelegate: onScreenTurnedOn()\r",
        "12-17 19:31:36.264  1795  1825 I WindowManger_keyguard: onScreenTurnedOn()\r",
        "12-17 19:31:36.264  1795  1825 D DisplayPowerController: Display ready!\r",
        "12-17 19:31:36.264  1795  1825 D DisplayPowerController: Finished business...\r",
        "12-17 19:31:36.264  2852  3328 D KeyguardService: Caller checkPermission fail\r",
        "12-17 19:31:36.264  2852  3328 D KeyguardService: KGSvcCall onScreenTurnedOn.\r",
        "12-17 19:31:36.264  2852  3328 D KeyguardViewMediator: notifyScreenTurnedOn\r",
        "12-17 19:31:36.265  2852  2852 D KeyguardViewMediator: handleNotifyScreenTurnedOn\r",
        "12-17 19:31:36.265  2852  2852 I PhoneStatusBar: onScreenTurnedOn\r",
        "12-17 19:31:36.265  2852  2852 D KGWallpaper_Magazine: getNextIndex: 0; from 5 to 5; size: 44\r",
        "12-17 19:31:36.265  2852  2852 I HwLockScreenReporter: report msg is :{picture: Deepwater-05-2.3.001-bigpicture_05_8.jpg}\r",
        "12-17 19:31:36.265  2852  2852 W HwLockScreenReporter: report result = falsereport type:162 msg:{picture: Deepwater-05-2.3.001-bigpicture_05_8.jpg, channelId: 05}\r",
        "12-17 19:31:36.265  2852  2852 I OucScreenOnCounter: Screen already turned on at: 1481974212\r",
        "12-17 19:31:36.267  5224  5283 I SendBroadcastPermission: action:android.com.huawei.bone.NOTIFY_SPORT_DATA, mPermissionType:0\r",
        "12-17 19:31:36.270  1795 16500 I HwActivityManagerService: Split enqueueing broadcast [callerApp]:ProcessRecord{580cfb2 5224:com.huawei.health:DaemonService/u0a99}\r",
        "12-17 19:31:36.271  2852  2852 I EventCenter: EventCenter Get :android.com.huawei.bone.NOTIFY_SPORT_DATA\r",
        "12-17 19:31:36.275  7741  7741 D Mms_TX_NOTIFY: Get no-perm notification callback android.intent.action.SCREEN_ON\r",
        "12-17 19:31:36.275  7741  7741 D Mms_TX_NOTIFY: ScreenState present\r",
        "12-17 19:31:36.275  5224  5283 I Step_HSNH: 20002302|upDateHealthNotification()|89|2.98|4180\r",
        "12-17 19:31:36.276  2883  2996 I HwSystemManager: ITrafficInfo:ITrafficInfo create 301updateBytes = 1769320345\r",
        "12-17 19:31:36.278  5224  5283 I Step_HSNH: 20002302|rebuild notification\r",
        "12-17 19:31:36.279  2852  2925 I EventCenter: ContentChange for slot: 1\r",
        "12-17 19:31:36.279  2852  2852 I HwBrightnessController: onChange selfChange:false uri.toString():content://settings/system/screen_auto_brightness mIsObserveAutoBrightnessChange:true\r",
        "12-17 19:31:36.279  1795  1825 D FpDataCollector: case xxx, not a fingerprint unlock \r",
        "12-17 19:31:36.280  1795  1825 D PowerManagerService: ready=true,policy=3,wakefulness=1,wksummary=0x11,uasummary=0x1,bootcompleted=true,boostinprogress=false,waitmodeenable=false,mode=true,manual=33,auto=-1,adj=0.0userId=0\r",
        "12-17 19:31:36.280  1795  1825 I PowerManager_screenOn: PowerManagerNotifier onWakefulnessChangeFinished mInteractiveChanging=true, mInteractive=true\r",
        "12-17 19:31:36.280  2852  2852 I HwBrightnessUtils: APS brightness=20.0,ConvertToPercentage=0.21667233\r",
        "12-17 19:31:36.280  2852  2852 I HwBrightnessUtils:  getSeekBarProgress isAutoMode:true current brightness:20 percentage:0.21667233\r",
        "12-17 19:31:36.280  2852  2852 I HwBrightnessController: updateSlider1 seekBarProgress:2167\r",
        "12-17 19:31:36.280  2852  2852 I HwBrightnessController: updateSlider2 seekBarProgress:2167\r",
        "12-17 19:31:36.280  2852  2852 I ToggleSlider:  mSeekListener onProgressChanged progress:2167 fromUser:false\r",
        "12-17 19:31:36.281  2852  2852 I ToggleSlider:  mSeekListener onProgressChanged progress:2167 fromUser:false\r",
        "12-17 19:31:36.282  3626  3753 I LogCollectService: msg = 103 received\r",
        "12-17 19:31:36.283  1795 11747 I NotificationManager: enqueueNotificationInternal: pkg=com.huawei.health id=10010 notification=Notification(pri=0 contentView=null vibrate=null sound=null defaults=0x0 flags=0x2 color=0x00000000 vis=PRIVATE)\r",
        "12-17 19:31:36.284  1795  1795 I NotificationManager: enqueueNotificationInternal: n.getKey = 0|com.huawei.health|10010|null|10099\r",
        "12-17 19:31:36.285  1795  2750 D HW_DISPLAY_EFFECT: presently, hw_update_color_temp_for_rg_led interface not achieved.\r",
        "12-17 19:31:36.285  3466  3466 I Contacts: DialpadFragment mBroadcastReceiver action:android.intent.action.SCREEN_ON\r",
        "12-17 19:31:36.289  3608  3608 D InCall  : InCallActivity - mScreenOnReceiver mCallEndOptionsDialog = null\r",
        "12-17 19:31:36.295  1795  1795 V NotificationService: disableEffects=null canInterrupt=false once update: false\r",
        "12-17 19:31:36.297  2852  2852 I StatusBar: onNotificationPosted: StatusBarNotification(pkg=com.huawei.health user=UserHandle{0} id=10010 tag=null key=0|com.huawei.health|10010|null|10099: Notification(pri=0 contentView=null vibrate=null sound=null defaults=0x0 flags=0x62 color=0x00000000 vis=PRIVATE)) important=2, post=1481974296283, when=1481531589202, vis=0, userid=0\r",
        "12-17 19:31:36.297  2852  2852 D StatusBar: updateNotification(StatusBarNotification(pkg=com.huawei.health user=UserHandle{0} id=10010 tag=null key=0|com.huawei.health|10010|null|10099: Notification(pri=0 contentView=null vibrate=null sound=null defaults=0x0 flags=0x62 color=0x00000000 vis=PRIVATE)))\r",
        "12-17 19:31:36.298  2852  2852 D HwCust  : Create obj success use class android.app.HwCustNotificationImpl\r",
        "12-17 19:31:36.299  2852  2852 I StatusBarIconView: updateTint: tint=0\r",
        "12-17 19:31:36.300  2852  2852 D StatusBar: No peeking: unimportant notification: 0|com.huawei.health|10010|null|10099\r",
        "12-17 19:31:36.301  2852  2852 D StatusBar: applyInPlace=true shouldPeek=false alertAgain=true\r",
        "12-17 19:31:36.301  2852  2852 I NotificationGroupManager: onEntryUpdated:0|com.huawei.health|10010|null|10099\r",
        "12-17 19:31:36.301  2852  2852 I NotificationGroupManager: onEntryAdded:0|com.huawei.health|10010|null|10099, group=0|com.huawei.health|10010|null|10099\r",
        "12-17 19:31:36.301  2852  2852 D StatusBar: reusing notification for key: 0|com.huawei.health|10010|null|10099\r",
        "12-17 19:31:36.301  2852  2852 D HwCust  : Create obj success use class android.app.HwCustNotificationImpl\r",
        "12-17 19:31:36.301  2852  2852 D HwCust  : Create obj success use class android.app.HwCustNotificationImpl\r",
        "12-17 19:31:36.302  2852  2852 I StatusBarIconView: updateTint: tint=0\r",
        "12-17 19:31:36.304 16628 16628 I TotemWeather: RetryTaskController:mTaskList is null\r",
        "12-17 19:31:36.311  2852  2852 I HwPhoneStatusBar: updateNotificationShade\r",
        "12-17 19:31:36.311  2852  2852 I PhoneStatusBar: updateNotificationShade\r",
        "12-17 19:31:36.311  2852  2852 I PhoneStatusBar: removeNotificationChildren\r",
        "12-17 19:31:36.311  2852  2852 I HwNotificationIconAreaController: showNotificationAll\r",
        "12-17 19:31:36.313 31949 31967 I PushService: main{1} PushService.onStartCommand(PushService.java:87) Push Service Start by  userEvent\r",
    ]
)
```

In the {mod}`ak.str` module there is the {func}`ak.str.extract_regex` function. This function decomposes an array of strings into an array of records, where each field of the newly created records corresponds to a named group in the regular expression. Let's define a regular expression to match our log

```{code-cell} ipython3
pattern = (
    # Timestamp
    r"(?P<datetime>\d\d-\d\d\s\d\d:\d\d:\d\d)\."
    # Fractional seconds
    r"(?P<datetime_frac>\d\d\d)\s\s"
    # Unknown integers
    r"(?P<i0>\d\d\d\d)\s\s"
    r"(?P<i1>\d\d\d\d)\s"
    # String category
    r"(?P<category>\w)\s"
    # String message
    r"(?P<message>.*)"
)
```

Does this match the first line?

```{code-cell} ipython3
lines[0]
```

Let's use the {mod}`re` module to use the above pattern to parse this line

```{code-cell} ipython3
import re

match = re.match(pattern, lines[0])
match.groupdict()
```

Let's now apply {func}`ak.str.extract_regex` to our array of lines using this pattern

```{code-cell} ipython3
structured = ak.str.extract_regex(lines, pattern)
structured
```

The type of the `structured` record is an "optional record of optional fields". This is because both the match itself can fail (producing the outer option), or the inner groups may be missing (producing the inner options). If we know that all groups should succeed or all groups should fail, then we can lift the inner options outside the record. To do this, we need to decompose the record, and rebuild it with `ak.zip` which provides a special `optiontype_outside_record` argument.

```{code-cell} ipython3
fields = ak.fields(structured)
contents = ak.unzip(structured)

result = ak.zip(dict(zip(fields, contents)), optiontype_outside_record=True)
result
```
