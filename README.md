<div align="center">
  <h1>Depict - share mental models better</h1>
</div>

<div align="center">
  <!-- Crates version -->
  <a href="https://crates.io/crates/depict">
    <img src="https://img.shields.io/crates/v/depict.svg?style=flat-square"
    alt="Crates.io version" />
  </a>
  <!-- Downloads -->
  <a href="https://crates.io/crates/depict">
    <img src="https://img.shields.io/crates/d/depict.svg?style=flat-square"
      alt="Download" />
  </a>
  <!-- docs -->
  <a href="https://docs.rs/depict">
    <img src="https://img.shields.io/badge/docs-latest-blue.svg?style=flat-square"
      alt="docs.rs docs" />
  </a>
  <!-- CI
  <a href="https://github.com/mstone/depict/actions">
    <img src="https://github.com/mstone/depict/actions/workflows/main.yml/badge.svg"
      alt="CI status" />
  </a> -->
  <!-- Discord -->
  <a href="https://discord.gg/UpWYZ5dN">
    <img src="https://img.shields.io/discord/973591045881360414.svg?logo=discord&style=flat-square" alt="Discord Link" />
  </a>
</div>

*Depict* helps people establish and validate shared mental models of complex systems and situations via pictures drawn from shorthand notes: ([demo](https://mstone.info/depict/))

[![An image, linking to a free hosted version of depict, of a model of a software development process involving a government agency, a contracted developer, an app, outside stakeholders, funders, and various feedback loops](https://raw.githubusercontent.com/mstone/depict/main/doc/agency.gif)](https://mstone.info/depict/)

<details>
<summary><i>agency situation shorthand notes</i></summary>
<pre>
agency [ priorities ]
developer [ design ]
agency developer: approve release,
developer app: release
developer code: update
agency developer: report issue,
agency app: /review design
stakeholders word: /review
funders agency: money, deliverables, timelines / grant application, renegotiation
developer json
word excel json code app -: _ : _ : _ : _
developer json: defines
agency developer: funding
agency word: / review
agency excel: edit
agency app: / test results,
agency app: test
agency developer: / report issue,
agency developer: prioritize
stakeholders agency: propose
</pre>
</details>

<!-- [![Depict live demo, showing a model of a microwave](https://raw.githubusercontent.com/mstone/depict/main/doc/microwave.gif)](https://mstone.info/depict/) -->

People who work with complex systems and situations often need to establish and validate shared mental models with partners in order to demonstrate understanding, build trust, and to set everyone up for success in future conversations.

Often, this process of establishing and validating shared mental models takes place by interviewing knowledgable people, taking notes, synthesizing notes into candidate pictures, and then reviewing the candidate pictures with stakeholders until everyone agrees that a satisfactory picture is available.

Unfortunately, many people find it hard to make these pictures in realtime as part of the interviewing process, let alone to make them legibly and comfortably. Specifically, conventional drawing tools often break the "flow" of the interview by requiring too much attention from the interviewer to use in realtime. People also often struggle to uncross arrows or to keep parts of their drawing from colliding, especially while editing text labels. Finally, the resulting drawings are brittle and are usually not meaningfully versioned or versionable such that people have trouble reusing and maintaining the resulting drawings over time.

*Depict* can drastically improve this "interview - record - synthesize - review" loop by:
* freeing interviewers' attention to improve and validate their understanding in realtime while maintaining flow
* enabling interviewers to concisely describe the players and interactions present in complex systems and situations via a shorthand notation specially developed for this purpose
* automatically drawing pretty, legible, maintainable pictures of models described by shorthand notes in realtime with minimal fuss, and
* automatically producing a transcript of the shared validated mental model developed so far which analysts can easily manipulate, version, reuse, and maintain.

## Getting *depict*

*depict* is available for free online at <https://mstone.info/depict/>.

Alternately, on macOS and Linux, you can build and run *depict* locally using [nix](https://nixos.org/nix/) with flakes enabled to run:

```bash
nix run github:mstone/depict#desktop
```

This should produce a window similar to the one shown in the screenshot above.

(For more information on how to install and use nix, see <https://zero-to-nix.com> and <https://mstone.info/posts/nix-tutorial/>).

## Using *depict*

*Depict* helps people establish and validate shared mental models with partners by automating the process of drawing pictures of situations involving complex interactions from shorthand notes such as might be recorded by an interviewer or an analyst on a video-call (possibly screensharing *depict* to enable other participants to review and help improve the interview or analysis team's developing understanding).

In these notes, each line of input describes a new part of the situation (system) to be drawn.

In the resulting drawing, processes can be ordered vertically (`a b`), horizontally (`a b -`), or via nesting (`a [ b ]`).

Additionally, interactions between processes can be shown with an arrow labeled in the "forward" (`a b: interaction`) direction from *a* to *b* or with an arrow labeled in the reverse (`a b: / interaction`) direction on either horizontal and vertical arrows.

(In the convention which this shorthand was invented to describe, downward-directed arrows represent "control actions" or "authority" of one player over another, upward arrows represent "feedback", rightward arrows represent "requests" between peers, leftward arrows represent "results" or "replies", and nesting represents how interacting parts can be abstracted, how higher-level conceptual processes can be decomposed, or the fate-sharing relationship between "platforms" and the processes they host.)

For example:

```
person microwave food: open, start, stop / beep : heat
person food: eat
```

says:

* there are `person`, `microwave`, and `food` boxes,
* `person` acts on `microwave`, `microwave` acts on `food`, and these interactions should vertically order these boxes
* in the space between `person` and `microwave`, there should be a downward arrow with three action labels, `open`, `start`, and `stop`, and an upward arrow with one feedback, `beep`,
* in the space between `microwave` and `food`, there is one action, `heat`.
* finally, there is also a direct relationship between `person` and `food` consisting of the action: `eat`.

For more detailed examples, please see these articles:

* [Introducing *depict](https://mstone.info/posts/introducing-depict/)
* [Real-world system depictions with *depict*](https://mstone.info/posts/real-world-system-depictions/)

## Syntax

*depict* offers an inline Syntax Guide with short examples of the *depict* shorthand input format.

Slightly formally though, the current *depict* input language roughly consists of:

| production  |   | syntax                                                  |
|-------------|---|---------------------------------------------------------|
|abbreviation |::=| *name* **:** *expr*
|relations    |::=| *name* *name* ... **[-]** (**:** *labels* (**/** */ *labels*)?)*
|labels       |::=| *label*... for single-word labels
|labels       |::=| *label* (**,** *label*)* for multi-word labels
|nesting      |::=| **[** *model* **]**
|alternatives |::=| **{** *model* **}**


## License

This project is licensed under the [MIT license].

[MIT license]: https://github.com/mstone/depict/blob/main/LICENSE

## Contribution

Unless you explicitly state otherwise, any contribution you intentionally submit for inclusion in *depict* shall be licensed as MIT without any additional terms or conditions.
