use depict::graph_drawing::error::{Error};
use depict::graph_drawing::geometry::{*};
use depict::graph_drawing::index::{LocSol, VerticalRank};
use depict::graph_drawing::layout::{*};

use std::borrow::Cow;
use std::fs::read_to_string;
use std::env::args;

use indoc::indoc;
use miette::{Diagnostic, NamedSource, Result};


pub fn tikz_escape(s: &str) -> String {
    s
        .replace('$', "\\$")
        .replace("\\n", "\\\\")
}

pub fn render<'s>(data: String) -> Result<(), Error> {
    let render_cell = depict::graph_drawing::frontend::render(Cow::Owned(data))?;
    let depiction = render_cell.borrow_dependent();
    
    let rs = &depiction.geometry_solution.rs;
    let ls = &depiction.geometry_solution.ls;
    let ss = &depiction.geometry_solution.ss;
    let all_locs = &depiction.geometry_problem.all_locs;
    let sol_by_loc = &depiction.geometry_problem.sol_by_loc;
    let node_to_loc = &depiction.layout_problem.node_to_loc;
    let loc_to_node = &depiction.layout_problem.loc_to_node;
    let vert_node_labels = &depiction.vcg.vert_node_labels;
    let vert_edge_labels = &depiction.vcg.vert_edge_labels;
    let hops_by_edge = &depiction.layout_problem.hops_by_edge;
    let sol_by_hop = &depiction.geometry_problem.sol_by_hop;
    let condensed = &depiction.cvcg.condensed;

    // std::process::exit(0);

    let base_width_scale = 1. / rs[LocSol(0)];
    let width_scale = 0.9;
    println!("{}", indoc!(r#"
    \documentclass[tikz,border=5mm]{standalone}
    \usetikzlibrary{graphs,graphdrawing,quotes,arrows.meta,calc,backgrounds,decorations.markings}
    \usegdlibrary{layered}
    \begin{document}
    
    \tikz[align=left, decoration={markings, mark=at position 0.5 with {\fill[red] (0, 0) circle (1pt);}}] {"#));

    for (loc, node) in loc_to_node.iter() {   
        let (ovr, ohr) = loc;
        let n = sol_by_loc[&(*ovr, *ohr)];

        let lpos = ls[n] * base_width_scale;
        let rpos = rs[n] * base_width_scale;

        let vpos = -1.5 * (ovr.0 as f64);
        let hpos = 10.0 * ((lpos + rpos) / 2.0);
        let width = 10.0 * width_scale * (rpos - lpos);

        match node {
            Obj::Node(vl) => {
                if vl == "root" {
                    continue
                }
                println!(indoc!(r#"
                    \node[minimum width = {}cm, fill=white, fill opacity=0.9, draw, text opacity=1.0]({}) at ({}, {}) {{{}}};"#), 
                    width, vl, hpos, vpos, vert_node_labels[&*vl]);
            },
            Obj::Hop(_, vl, wl) => {
                if vl == "root" {
                    continue
                }
                let hn = sol_by_hop[&(*ovr, *ohr, vl.clone(), wl.clone())];
                let spos = ss[hn] * base_width_scale;
                let hpos = 10.0 * spos;
                println!(indoc!(r#"
                    \draw [fill, black] ({}, {}) circle (0.5pt);
                    \node[](aux_{}_{}) at ({}, {}) {{}};"#), 
                    hpos, vpos, ovr, ohr, hpos, vpos);
            },
            Obj::Border(border) => {
                todo!()
            }
        }
    }

    for cer in condensed.edge_references() {
        for (vl, wl, ew) in cer.weight().iter() {
            if vl == "root" {
                continue
            }
            let label_text = vert_edge_labels.get(vl).and_then(|dsts| dsts.get(wl).and_then(|rels| rels.get(ew)))
                .map(|v| v.join("\n"))
                .unwrap_or_else(|| ew.to_string());

            let (ovr, ohr) = node_to_loc[&Obj::Node(vl.clone())];
            let (ovrd, ohrd) = node_to_loc[&Obj::Node(wl.clone())];

            let snv = sol_by_hop[&(ovr, ohr, vl.clone(), wl.clone())];
            let snw = sol_by_hop[&(ovrd, ohrd, vl.clone(), wl.clone())];

            let sposv = ss[snv] * base_width_scale;
            let sposw = ss[snw] * base_width_scale;

            let nnv = sol_by_loc[&(ovr, ohr)];
            let nnw = sol_by_loc[&(ovrd, ohrd)];

            let lposv = ls[nnv] * base_width_scale;
            let lposw = ls[nnw] * base_width_scale;

            let rposv = rs[nnv] * base_width_scale;
            let rposw = rs[nnw] * base_width_scale;

            let src_width = rposv - lposv;
            let dst_width = rposw - lposw;

            let bundle_src_frac = ((((sposv - lposv) / src_width) - 0.5) / width_scale) + 0.5;
            let bundle_dst_frac = ((((sposw - lposw) / dst_width) - 0.5) / width_scale) + 0.5;

            let arr_src_frac = match ew {
                x if x == "actuates" => (bundle_src_frac) - (0.025 / src_width),
                x if x == "senses" => (bundle_src_frac) + (0.025 / src_width),
                _ => (bundle_src_frac),
            };
            let arr_dst_frac = match ew {
                x if x == "actuates" => bundle_dst_frac - (0.025 / dst_width),
                x if x == "senses" => bundle_dst_frac + (0.025 / dst_width),
                _ => bundle_dst_frac,
            };

            let hops = &hops_by_edge[&(vl.clone(), wl.clone())];
            eprintln!("vl: {}, wl: {}, hops: {:?}", vl, wl, hops);

            let dir = match ew {
                x if x == "actuates" || x == "rides" => "-{Stealth[]}",
                x if x == "senses" => "{Stealth[]}-",
                _ => "-",
            };

            let anchor = match ew {
                x if x == "actuates" => "north east",
                x if x == "senses" => "south west",
                _ => "south east",
            };

            match hops.len() {
                0 => { unreachable!(); }
                1 => {
                    println!(indoc!(r#"
                        \draw [{}] ($({}.south west)!{}!({}.south east)$)        to[]  node[scale=0.8, anchor={}, fill=white, fill opacity = 0.8, text opacity = 1.0, draw, ultra thin] {{{}}} ($({}.north west)!{}!({}.north east)$);"#),
                        dir, vl, arr_src_frac, vl, anchor, label_text, wl, arr_dst_frac, wl    
                    );
                },
                2 => {
                    let (lvl, (_mhr, nhr)) = hops.iter().next().unwrap();
                    let (ovr, ohr) = (*lvl+1, nhr);
                    // BUG? -- intermediate node needs to be at aux_{ovr}_{ohr}?
                    println!(indoc!(r#"
                        \draw [rounded corners, {}] ($({}.south west)!{}!({}.south east)$) -- node[scale=0.8, anchor={}, fill=white, fill opacity = 0.8, text opacity = 1.0, draw, ultra thin] {{{}}} at ({},{}) -- ($({}.north west)!{}!({}.north east)$);"#),
                        dir, vl, arr_src_frac, vl, anchor, label_text, ovr, ohr, wl, arr_dst_frac, wl    
                    );
                },
                max_levels => {
                    print!(indoc!(r#"\draw [rounded corners, {}] ($({}.south west)!{}!({}.south east)$)"#), 
                        dir, vl, arr_src_frac, vl);
                    let mid = max_levels / 2;
                    let mut mid_ovr = 0;
                    let mut mid_ohr = 0;
                    let mut mid_ovrd = 0;
                    let mut mid_ohrd = 0;
                    for (n, hop) in hops.iter().enumerate() {
                        if n < max_levels-1 {
                            let (lvl, (mhr, nhr)) = hop;
                            let (ovr, ohr) = (*lvl+1, nhr);
                            // let (ovr, ohr) = (lvl, mhr);
                            println!("% HOP {} {:?}", n, hop);
                            print!(r#" -- (aux_{}_{}.center)"#, ovr, ohr);
                            if n == mid {
                                mid_ovr = lvl.0;
                                mid_ohr = mhr.0;
                                mid_ovrd = (*lvl+1).0;
                                mid_ohrd = nhr.0;
                            }
                        }
                    }
                    println!(indoc!(r#" -- ($({}.north west)!{}!({}.north east)$);"#), wl, arr_dst_frac, wl);
                    println!(indoc!(r#"\node[scale=0.8, anchor={}, fill=white, fill opacity = 0.8, text opacity = 1.0] (mid_{}_{}_{}) at ($(aux_{}_{})!0.5!(aux_{}_{})$) {{{}}};"#), 
                        anchor, vl, wl, ew, mid_ovr, mid_ohr, mid_ovrd, mid_ohrd, label_text);
                },
            }
        }
    }

    use rand::Rng;
    let mut rng = rand::thread_rng();

    for LocRow{ovr, ohr, shr: _, loc, ..} in all_locs.iter() {
        if *ovr == VerticalRank(0) {
            continue
        }
        let n = sol_by_loc[&(*ovr, *ohr)];

        let lpos = ls[n] * base_width_scale;
        let rpos = rs[n] * base_width_scale;

        let vpos = -1.5 * (ovr.0 as f64) + 0.5 - rng.gen_range(0.25..0.75);
        let hpos = 10.0 * ((lpos + rpos) / 2.0);
        let hposl = 10.0 * lpos;
        let hposr = 10.0 * rpos;

        let loc_color = match loc {
            Obj::Node(_) => "red",
            Obj::Hop(_, _, _) => "blue",
            Obj::Border(_) => "pink"
        };
        let loc_str = match loc {
            Obj::Node(vl) => vl.to_string(),
            Obj::Hop(_, vl, wl) => format!("{}{}", vl.chars().next().unwrap(), wl.chars().next().unwrap()),
            Obj::Border(border) => todo!(),
        };

        println!(indoc!(r#"%\draw [{}] ({}, {}) circle (1pt);"#), loc_color, hpos, vpos);
        println!(indoc!(r#"%\draw [fill,violet] ({}, {}) circle (0.5pt);"#) , hposl, vpos);
        println!(indoc!(r#"%\draw [fill,orange] ({}, {}) circle (0.5pt);"#), hposr, vpos);
        println!(indoc!(r#"%\draw [--] ({},{}) -- ({}, {});"#), hposl, vpos, hposr, vpos);
        println!(indoc!(r#"%\node[scale=0.5, anchor=south west] at ({}, {}) {{{}}};"#), hpos, vpos, loc_str);
    }

    for ((lvl, _mhr, vl, wl), n) in sol_by_hop.iter() {
        let spos = ss[*n] * base_width_scale;

        // let vpos = -1.5 * (*lvl as f64) - 0.5 + rng.gen_range(0.5..1.0);
        let vpos = -1.5 * (lvl.0 as f64);
        let hpos = 10.0 * (spos);// + rng.gen_range(0.0..0.25);
        
        println!(indoc!(r#"%\draw [fill, black] ({}, {}) circle (1pt);"#), hpos, vpos);
        println!(indoc!(r#"%\node[scale=0.5, anchor=south east] at ({}, {}) {{{}{}}};"#), hpos, vpos, vl.chars().next().unwrap(), wl.chars().next().unwrap());
    }


    // let l = -3;
    // let r = 12;
    // let t = 7;
    // let b = -1;
    // println!(indoc!(r#"
    //     \scope[on background layer]
    //     \draw[help lines,very thin,step=1] ({}.2,{}.2) grid ({}.2,{}.2);
    //     \foreach \x in {{{},...,{}}} {{
    //     \foreach \y in {{{},...,{}}} {{
    //         \draw [fill, black] (\x, \y) circle (0.5pt); 
    //         \node[scale=0.5, anchor=south east] at (\x, \y) {{\x,\y}};
    //     }}
    //     }}
    //     \endscope
    // "#), l, b, r, t, l, r, b, t);
    println!(indoc!(r#"
        }}
        \end{{document}}
    "#));

    Ok(())
}

#[derive(Debug, Diagnostic, thiserror::Error)]
#[diagnostic(code(depict::parse_error))]
pub enum TikzError {
    #[error("parse error")]
    ParseError {
        #[source_code]
        src: NamedSource,

        #[label = "Unexpected token"]
        span: std::ops::Range<usize>,

        text: String,
    },
    #[error("io error")]
    IoError {
        #[from] source: std::io::Error
    },
}

pub fn main() -> Result<()> {
    for path in args().skip(1) {
        let data = read_to_string(path.clone())
            .map_err(|e| TikzError::IoError{source: e})?;
        
        render(data)?;
    }
    Ok(())
}
