// clang -g -Wall -framework Cocoa -o main main.m
use cocoa::{self, foundation::{NSAutoreleasePool, NSProcessInfo, NSString, NSRect, NSPoint, NSSize}, base::{nil, selector, NO, id}, appkit::{NSApplication, NSMenu, NSMenuItem, NSWindow, NSBackingStoreType::NSBackingStoreBuffered, NSWindowStyleMask, NSRunningApplication, NSApplicationActivationOptions::NSApplicationActivateIgnoringOtherApps, NSApp, NSApplicationActivationPolicy::NSApplicationActivationPolicyRegular}};
use objc::{class, declare::ClassDecl, runtime::{Object, Sel}, sel, sel_impl, msg_send};

// https://ar.al/2018/09/17/workaround-for-unclickable-app-menu-bug-with-window.makekeyandorderfront-and-nsapp.activate-on-macos/
// https://stackoverflow.com/questions/33345686/cocoa-application-menu-bar-not-clickable

// -(void)applicationWillFinishLaunching:(NSNotification *)aNotification
// {
//   [CustomApplication sharedApplication];
//   [CustomApplication setUpMenuBar];
//   [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
// }

// -(void)applicationDidFinishLaunching:(NSNotification *)notification
// {
//   [CustomApplication sharedApplication];

//   [NSApp activateIgnoringOtherApps:YES];
// }



pub fn main() {
    unsafe {
        let _pool = NSAutoreleasePool::new(nil);
        // let app = NSApplication::sharedApplication(nil);
        let app = NSApp();

        let superclass = class!(NSObject);
        let mut decl = ClassDecl::new("MyApplicationDelegate", superclass).unwrap();

        extern fn application_will_finish_launching(_: &Object, _: Sel, _: id) {
            unsafe {       
                NSApp().setActivationPolicy_(NSApplicationActivationPolicyRegular);
            }
        }
        extern fn application_did_finish_launching(_: &Object, _: Sel, _: id) {
            unsafe {
                let current_app = NSRunningApplication::currentApplication(nil);
                current_app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps);
            }
        }

        decl.add_method(sel!(applicationWillFinishLaunching:),
            application_will_finish_launching as extern fn(&Object, Sel, id));

        decl.add_method(sel!(applicationDidFinishLaunching:),
            application_did_finish_launching as extern fn(&Object, Sel, id));
        
        let delegate_class = decl.register();
        let delegate_object: id = msg_send![delegate_class, new];
        delegate_object.autorelease();

        NSApp().setDelegate_(delegate_object);

        let menubar_title = NSString::alloc(nil).init_str("").autorelease();
        let menubar = NSMenu::new(nil).initWithTitle_(menubar_title).autorelease();
        let app_menu_item = NSMenuItem::new(nil).autorelease();
        menubar.addItem_(app_menu_item);
        app.setMainMenu_(menubar);

        let app_menu = NSMenu::new(nil).autorelease();
        let app_name = NSProcessInfo::processInfo(nil).processName();
        let quit_title = NSString::alloc(nil).init_str("Quit ").stringByAppendingString_(app_name);
        let quit_action = selector("terminate:");
        let quit_key = NSString::alloc(nil).init_str("q");
        let quit_item = NSMenuItem::alloc(nil)
            .initWithTitle_action_keyEquivalent_(quit_title, quit_action, quit_key)
            .autorelease();
        app_menu.addItem_(quit_item);
        app_menu_item.setSubmenu_(app_menu);

        let rect = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(400., 200.));
        let style = NSWindowStyleMask::NSTitledWindowMask;
        let backing = NSBackingStoreBuffered;
        let defer = NO;
        let window = NSWindow::alloc(nil)
            .initWithContentRect_styleMask_backing_defer_(rect, style, backing, defer)
            .autorelease();
        window.cascadeTopLeftFromPoint_(NSPoint::new(20., 20.));
        window.center();
        let title = NSString::alloc(nil).init_str("Hello World!");
        window.setTitle_(title);
        window.makeKeyAndOrderFront_(nil);
        app.run();
    }
}

    