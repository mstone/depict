use diagrams::parser::{parse, Ident};
use diagrams::graph_drawing::*;
use diagrams::render::filter_fact;
use druid::widget::Scroll;
use druid::kurbo::{PathEl};
use druid::piet::RenderContext;
use petgraph::dot::Dot;
use std::fs::read_to_string;
use std::env::args;
use std::io;

use druid::{self, AppLauncher, WindowDesc, Data, Size, Point, Color, LifeCycle, WidgetPod, LocalizedString, MenuItem, commands, SysMods, Menu};
use druid::{widget::{Label}, Widget};

#[derive(Clone, Data, Default)]
pub struct AppState(String);

#[derive(Default)]
pub struct AppWidget<T> {
    texts: Vec<(WidgetPod<T,Box<dyn Widget<T>>>, Point)>,
    arrows: Vec<Vec<PathEl>>,
}

impl Widget<AppState> for AppWidget<AppState> {
    fn event(&mut self, ctx: &mut druid::EventCtx, event: &druid::Event, data: &mut AppState, env: &druid::Env) {
        for text in self.texts.iter_mut() {
            text.0.event(ctx, event, data, env)
        }
    }

    fn lifecycle(&mut self, ctx: &mut druid::LifeCycleCtx, event: &druid::LifeCycle, data: &AppState, env: &druid::Env) {
        if matches!(event, LifeCycle::WidgetAdded) {
            let v = parse(&data.0[..]).unwrap().1;
            let mut ds = filter_fact(v.iter(), &Ident("draw"));
            let draw = ds.next().unwrap();

            let (vert, v_nodes, h_name) = calculate_vcg(&v, draw);

            eprintln!("VERT: {:?}", Dot::new(&vert));

            let (condensed, _) = condense(&vert);

            let roots = roots(&condensed);

            let paths_by_rank = rank(&condensed, &roots);

            let (locs_by_level, hops_by_level, hops_by_edge, loc_to_node, node_to_loc) = calculate_locs_and_hops(&condensed, &paths_by_rank);

            let solved_locs = minimize_edge_crossing(&locs_by_level, &hops_by_level);
            
            let (all_locs, all_hops0, all_hops, sol_by_loc, sol_by_hop) = calculate_sols(&solved_locs, &loc_to_node, &hops_by_level, &hops_by_edge);

            let (ls, rs, ss) = position_sols(&vert, &v_nodes, &hops_by_edge, &node_to_loc, &solved_locs, &all_locs, &all_hops0, &all_hops, &sol_by_loc, &sol_by_hop);
            
            // let width_scale = 0.9;
            let height_scale = 80.0;
            
            self.texts = vec![];

            for (loc, node) in loc_to_node.iter() {   
                let (ovr, ohr) = loc;
                let n = sol_by_loc[&(*ovr, *ohr)];

                let lpos = ls[n];
                let rpos = rs[n];

                let vpos = 80.0 * (*ovr as f64) + 100.0;
                let hpos = 1024.0 * ((lpos + rpos) / 2.0);
                // let width = 1024.0 * width_scale * (rpos - lpos);

                match node {
                    Loc::Node(vl) => {
                        let label = Label::new(h_name[*vl].clone());
                        let wp = WidgetPod::new(label).boxed();
                        self.texts.push((wp, Point::new(hpos, vpos)));
                    },
                    Loc::Hop(_, _vl, _wl) => {
                        // let hn = sol_by_hop[&(*ovr, *ohr, *vl, *wl)];
                        // let spos = ss[hn];
                        // let hpos = 10.0 * spos;
                    },
                }
            }

            self.arrows = vec![];

            for cer in condensed.edge_references() {
                for (vl, wl, _ew) in cer.weight().iter() {
                    let (ovr, ohr) = node_to_loc[&Loc::Node(*vl)];

                    let snv = sol_by_hop[&(ovr, ohr, *vl, *wl)];

                    let sposv = ss[snv];

                    let hops = &hops_by_edge[&(*vl, *wl)];
                    eprintln!("vl: {}, wl: {}, hops: {:?}", vl, wl, hops);

                    let mut path = vec![];
                    
                    for (n, hop) in hops.iter().enumerate() {
                        let (lvl, (_mhr, nhr)) = hop;
                        let hn = sol_by_hop[&(lvl+1, *nhr, *vl, *wl)];
                        let spos = ss[hn];
                        let hpos = 1024.0 * spos;

                        if n == 0 {
                            path.push(PathEl::MoveTo(Point::new(1024.0 * sposv, (*lvl as f64) * height_scale + 100.0)));
                        }
                        
                        path.push(PathEl::LineTo(Point::new(hpos, ((*lvl + 1) as f64) * height_scale + 100.0)));
                        
                    }

                    self.arrows.push(path);
                }
            }
        }
        for text in self.texts.iter_mut() {
            text.0.lifecycle(ctx, event, data, env)
        }
    }

    fn update(&mut self, ctx: &mut druid::UpdateCtx, _old_data: &AppState, data: &AppState, env: &druid::Env) {
        for text in self.texts.iter_mut() {
            text.0.update(ctx, data, env)
        }
    }

    fn layout(&mut self, ctx: &mut druid::LayoutCtx, bc: &druid::BoxConstraints, data: &AppState, env: &druid::Env) -> druid::Size {
        for (text, origin) in self.texts.iter_mut() {
            let size = text.layout(ctx, bc, data, env);
            let mut pos = origin.clone();
            pos.x -= size.width / 2.0;
            text.set_origin(ctx, data, env, pos);
        }
        let default_size = Size::new(1024., 768.);
        let size = bc.constrain(default_size);
        size
    }

    fn paint(&mut self, ctx: &mut druid::PaintCtx, data: &AppState, env: &druid::Env) {
        for (text, _origin) in self.texts.iter_mut() {
            text.paint(ctx, data, env);
        }
        let brush = ctx.solid_brush(Color::RED);
        for arrow in self.arrows.iter() {
            ctx.stroke(&arrow[..], &brush, 1.0);
        }
    }
}

pub fn build_ui() -> impl Widget<AppState> {
    Scroll::new(AppWidget{..Default::default()})
}

fn macos_application_menu<T: Data>() -> Menu<T> {
    // Menu::new("Hi")
    // druid::platform_menus::mac::menu_bar()
    Menu::new("WTF")
        .entry(druid::platform_menus::mac::application::default())
        .entry(druid::platform_menus::mac::file::default())
    // Menu::new(LocalizedString::new("macos-menu-application-menu"))
    //     .entry(MenuItem::new( LocalizedString::new("macos-menu-about-app") ).command( commands::SHOW_ABOUT ))
    //     .separator()
    //     .entry(
    //         MenuItem::new(
    //             LocalizedString::new("macos-menu-preferences"),
    //         )
    //         .command(
    //             commands::SHOW_PREFERENCES,
    //         )
    //         .hotkey(SysMods::Cmd, ",")
    //         .enabled(false),
    //     )
    //     .separator()
    //     .entry(Menu::new(LocalizedString::new("macos-menu-services")))
    //     .entry(
    //         MenuItem::new(
    //             LocalizedString::new("macos-menu-hide-app"),
    //         ).command(
    //             commands::HIDE_APPLICATION,
    //         )
    //         .hotkey(SysMods::Cmd, "h"),
    //     )
    //     .entry(
    //         MenuItem::new(
    //             LocalizedString::new("macos-menu-hide-others"),
    //         ).command(
    //             commands::HIDE_OTHERS,
    //         )
    //         .hotkey(SysMods::AltCmd, "h"),
    //     )
    //     .entry(
    //         MenuItem::new(
    //             LocalizedString::new("macos-menu-show-all"),
    //         ).command(
    //             commands::SHOW_ALL,
    //         )
    //         .enabled(false),
    //     )
    //     .separator()
    //     .entry(
    //         MenuItem::new(
    //             LocalizedString::new("macos-menu-quit-app"),
    //         ).command(
    //             commands::QUIT_APP,
    //         )
    //         .hotkey(SysMods::Cmd, "q"),
    //     )
}

pub fn render(contents: String) {
    let main_window = WindowDesc::new(build_ui())
        .title("STAMPEDE")
        .window_size((1050.0, 600.0))
        .menu(|_window_id, _data, _env| {
            eprintln!("MENU MENU MENU MENU");
            macos_application_menu()
        });
        // .menu(
        //     |_window_id, _data, _env| druid::menu::sys::mac::application::default()
        // );

    let initial_state = AppState(contents);

    AppLauncher::with_window(main_window)
        .log_to_console()
        // .localization_resources(vec!["builtin.ftl".into()], "./resources/i18n".into())
        .launch(initial_state)
        .expect("Failed to launch application");
}

pub fn main() -> io::Result<()> {
    for path in args().skip(1) {
        let contents = read_to_string(path)?;
        // println!("{}\n\n", &contents);
        render(contents);
    }
    Ok(())
}