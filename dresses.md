Hey, got a project I’d love your eyes on. I run the website for a formalwear store. Sherri Hill (big prom brand) only photographs a couple of colors per dress, but most styles come in 10 or more. The idea is to use AI to show the dress in every color on a model.

The catch is we can’t use their model, because of likeness rights. So the plan is to use their photo as a reference for the dress itself, plus a photo of the real dresses on a rack showing every color, and generate a new model wearing it.

It has to be accurate. These are $500 to $1,500 dresses covered in beads and pearls, and if the color or the beading is off, people send them back. I was thinking of testing it on a dress where we already have real photos of 3 colors, generating those 3 blind, and comparing.

Is that doable today at that level of detail?
Matt — 8:12 PM
Yeah I think so
Typheratrix — 8:12 PM
Here is an example. Would be for hundreds of dresses so cost might be a blocker. Don't know anything about that side of the house so thought I would reach out
/Users/matthewstovall/Documents/dev/Sandbox/dresses/dresses-model-wearing-close.jpeg
/Users/matthewstovall/Documents/dev/Sandbox/dresses/dresses-model-wearing-full.jpeg
/Users/matthewstovall/Documents/dev/Sandbox/dresses/dresses-multiple-hanging.jpeg
Matt — 8:15 PM
could prob use a mask on the dress, to preserve entirely, and generate the rest of the model wearing the drees around that mask
color changing should be doable with ideogram-4.5 and sampling the image with a modern llm to extract the colors
if all of that works should be less than $1 per dress
