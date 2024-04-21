import { Dolos } from "@dodona/dolos-lib";
import { File, Region } from "@dodona/dolos-core";

// Each of the two file paths is given by argument input
const files = process.argv.slice(2);
const dolos = new Dolos();
const report = await dolos.analyzePaths(files);

let temp = [];
let test_f;

for (const pair of report.allPairs()) {
    for (const fragment of pair.buildFragments()) {
        const right = fragment.rightSelection;
        temp.push(pair.rightFile.lines.slice(right.startRow, right.endRow+1).join("\n"))
    }
    test_f = pair.leftFile.content
}


const f1 = new File("file1.py", test_f);
const f2 = new File("file2.py", temp.join("\n"));

const subreport = await dolos.analyze([f1, f2]);

const similarity = subreport.getPair(f1, f2).similarity;
console.log(`Similarity: ${similarity}`);
