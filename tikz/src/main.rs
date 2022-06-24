use depict::graph_drawing::error::{Error};
use depict::graph_drawing::geometry::{*};
use depict::graph_drawing::graph::roots;
use depict::graph_drawing::layout::{*};
use depict::parser::{Fact, Parser, Token, Item};

use std::fs::read_to_string;
use std::env::args;

use indoc::indoc;
use logos::Logos;
use miette::{Diagnostic, NamedSource, Result};
use petgraph::dot::{Dot};


pub fn tikz_escape(s: &str) -> String {
    s
        .replace('$', "\\$")
        .replace("\\n", "\\\\")
}

pub fn render(v: Vec<Fact<&str>>) -> Result<(), Error> {

    let vcg = calculate_vcg(&v)?;
    let Vcg{vert, vert_vxmap: _, vert_node_labels, vert_edge_labels} = &vcg;

    eprintln!("VERT: {:?}", Dot::new(&vert));

    let Cvcg{condensed, condensed_vxmap: _} = condense(vert)?;

    let roots = roots(&condensed)?;

    let paths_by_rank = rank(&condensed, &roots)?;

    let layout_problem = calculate_locs_and_hops(&condensed, &paths_by_rank)?;
    let LayoutProblem{hops_by_level, hops_by_edge, loc_to_node, node_to_loc, ..} = &layout_problem;

    // std::process::exit(0);

    let (_crossing_number, solved_locs) = minimize_edge_crossing(&layout_problem)?;
    
    let geometry_problem = calculate_sols(&solved_locs, loc_to_node, hops_by_level, hops_by_edge);

    let GeometrySolution{ls, rs, ss, ..} = position_sols(&vcg, &layout_problem, &solved_locs, &geometry_problem)?;

    let GeometryProblem{all_locs, sol_by_loc, sol_by_hop, ..} = geometry_problem;

    // std::process::exit(0);

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

        let lpos = ls[n];
        let rpos = rs[n];

        let vpos = -1.5 * (ovr.0 as f64);
        let hpos = 10.0 * ((lpos + rpos) / 2.0);
        let width = 10.0 * width_scale * (rpos - lpos);

        match node {
            Loc::Node(vl) => {
                println!(indoc!(r#"
                    \node[minimum width = {}cm, fill=white, fill opacity=0.9, draw, text opacity=1.0]({}) at ({}, {}) {{{}}};"#), 
                    width, vl, hpos, vpos, vert_node_labels[*vl]);
            },
            Loc::Hop(_, vl, wl) => {
                let hn = sol_by_hop[&(*ovr, *ohr, *vl, *wl)];
                let spos = ss[hn];
                let hpos = 10.0 * spos;
                println!(indoc!(r#"
                    \draw [fill, black] ({}, {}) circle (0.5pt);
                    \node[](aux_{}_{}) at ({}, {}) {{}};"#), 
                    hpos, vpos, ovr, ohr, hpos, vpos);
            },
        }
    }

    for cer in condensed.edge_references() {
        for (vl, wl, ew) in cer.weight().iter() {
            let label_text = vert_edge_labels.get(vl).and_then(|dsts| dsts.get(wl).and_then(|rels| rels.get(ew)))
                .map(|v| v.join("\n"))
                .unwrap_or_else(|| ew.to_string());

            let (ovr, ohr) = node_to_loc[&Loc::Node(*vl)];
            let (ovrd, ohrd) = node_to_loc[&Loc::Node(*wl)];

            let snv = sol_by_hop[&(ovr, ohr, *vl, *wl)];
            let snw = sol_by_hop[&(ovrd, ohrd, *vl, *wl)];

            let sposv = ss[snv];
            let sposw = ss[snw];

            let nnv = sol_by_loc[&(ovr, ohr)];
            let nnw = sol_by_loc[&(ovrd, ohrd)];

            let lposv = ls[nnv];
            let lposw = ls[nnw];

            let rposv = rs[nnv];
            let rposw = rs[nnw];

            let src_width = rposv - lposv;
            let dst_width = rposw - lposw;

            let bundle_src_frac = ((((sposv - lposv) / src_width) - 0.5) / width_scale) + 0.5;
            let bundle_dst_frac = ((((sposw - lposw) / dst_width) - 0.5) / width_scale) + 0.5;

            let arr_src_frac = match *ew {
                "actuates" => (bundle_src_frac) - (0.025 / src_width),
                "senses" => (bundle_src_frac) + (0.025 / src_width),
                _ => (bundle_src_frac),
            };
            let arr_dst_frac = match *ew {
                "actuates" => bundle_dst_frac - (0.025 / dst_width),
                "senses" => bundle_dst_frac + (0.025 / dst_width),
                _ => bundle_dst_frac,
            };

            let hops = &hops_by_edge[&(*vl, *wl)];
            eprintln!("vl: {}, wl: {}, hops: {:?}", vl, wl, hops);

            let dir = match *ew {
                "actuates" | "rides" => "-{Stealth[]}",
                "senses" => "{Stealth[]}-",
                _ => "-",
            };

            let anchor = match *ew {
                "actuates" => "north east",
                "senses" => "south west",
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
        let n = sol_by_loc[&(*ovr, *ohr)];

        let lpos = ls[n];
        let rpos = rs[n];

        let vpos = -1.5 * (ovr.0 as f64) + 0.5 - rng.gen_range(0.25..0.75);
        let hpos = 10.0 * ((lpos + rpos) / 2.0);
        let hposl = 10.0 * lpos;
        let hposr = 10.0 * rpos;

        let loc_color = match loc {
            Loc::Node(_) => "red",
            Loc::Hop(_, _, _) => "blue",
        };
        let loc_str = match loc {
            Loc::Node(vl) => vl.to_string(),
            Loc::Hop(_, vl, wl) => format!("{}{}", vl.chars().next().unwrap(), wl.chars().next().unwrap()),
        };

        println!(indoc!(r#"%\draw [{}] ({}, {}) circle (1pt);"#), loc_color, hpos, vpos);
        println!(indoc!(r#"%\draw [fill,violet] ({}, {}) circle (0.5pt);"#) , hposl, vpos);
        println!(indoc!(r#"%\draw [fill,orange] ({}, {}) circle (0.5pt);"#), hposr, vpos);
        println!(indoc!(r#"%\draw [--] ({},{}) -- ({}, {});"#), hposl, vpos, hposr, vpos);
        println!(indoc!(r#"%\node[scale=0.5, anchor=south west] at ({}, {}) {{{}}};"#), hpos, vpos, loc_str);
    }

    for ((lvl, _mhr, vl, wl), n) in sol_by_hop.iter() {
        let spos = ss[*n];

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

    // println!("{}", "\n\n\n");
    // println!("{:?}", Dot::new(&vert));

    // use top-level "draw" fact to identify inline or top-level drawings to draw
    // resolve top-level drawings + use inline drawings to identify objects to draw to make particular drawings
    // use object facts to figure out directions + labels?

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
        // println!("{}\n\n", &contents);
        let mut p = Parser::new();
        let mut lex = Token::lexer(&data);
        while let Some(tk) = lex.next() {
            p.parse(tk)
                .map_err(|_| { 
                    TikzError::ParseError{
                        src: NamedSource::new(path.clone(), data.clone()), 
                        span: lex.span(), 
                        text: lex.slice().into()
                    }
                })?
        }

        let v: Vec<Item> = p.end_of_input().map_err(|_| { 
            TikzError::ParseError{
                src: NamedSource::new(path.clone(), data.clone()), 
                span: lex.span(), 
                text: lex.slice().into()
            }
        })?;

        eprintln!("PARSE {v:#?}");

        // let vcg = calculate_vcg2(&v)?;
    }
    Ok(())
}
