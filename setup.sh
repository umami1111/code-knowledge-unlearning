#!/bin/bash
mkdir tmp
cd tmp
curl -X GET "https://zenodo.org/api/records/7987148/files-archive" \
     -H "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8" \
     -H "Accept-Encoding: gzip, deflate, br, zstd" \
     -H "Accept-Language: en-GB,en;q=0.8" \
     -H "Connection: keep-alive" \
     -H "Cookie: 5569e5a730cade8ff2b54f1e815f3670=2b1e3011f08bc06d94801540527db312; session=5bf4c4e54ff99e49_6611fd9c.AEz19h0v--yMuxR6sKanKfzrGjU; csrftoken=eyJhbGciOiJIUzUxMiIsImlhdCI6MTcxMjQ1NTA2OCwiZXhwIjoxNzEyNTQxNDY4fQ.InVBQVVlVkliM3dkbjVWVjRzNkF3b2p3aG5uYk83enl1Ig.4y-MET57wmt15xr6Rl6t8G1DKLx5oiwh76IcESc-UxACkhyRkaYYhMqBIVujPqmWIAI0zUQM_gPKKH68VuNALQ" \
     -H "Host: zenodo.org" \
     -H "Referer: https://zenodo.org/records/7987148" \
     -H "Sec-Fetch-Dest: document" \
     -H "Sec-Fetch-Mode: navigate" \
     -H "Sec-Fetch-Site: same-origin" \
     -H "Sec-Fetch-User: ?1" \
     -H "Sec-GPC: 1" \
     -H "Upgrade-Insecure-Requests: 1" \
     -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36" \
     -H "sec-ch-ua: \"Brave\";v=\"123\", \"Not:A-Brand\";v=\"8\", \"Chromium\";v=\"123\"" \
     -H "sec-ch-ua-mobile: ?0" \
     -H "sec-ch-ua-platform: Windows" > files.zip
sudo apt update -y
sudo apt install -y unzip
unzip files.zip
cp -r Prompts_Original.zip ../Prompts
cp -r Source_Original.zip ../Source/
cd ../Prompts/
unzip Prompts_Original.zip
mv Prompts_Original/* ./
rm -r Prompts_Original
cd ../Source_Original
unzip Source_Original.zip
mv Source_Original/* ./
rm -r Source_Original
cd ../

sudo apt install -y python3.10-venv
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sudo apt install -y build-essential
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
nvm install node
nvm use node
npm install @dodona/dolos-lib
npm install tree-sitter-cpp@0.20
sudo apt install -y openjdk-17-jre-headless
sudo apt install -y openjdk-17-jdk-headless

# Note: Rerun if you see UnicodeEncodeError: 'ascii' codec can't encode characters in position 99-107: ordinal not in range(128)
# python3 Example_Parrot.py codeparrot/codeparrot-small prompts_agpl3_python_2023-03-27-21-21-29
# python3 model_eval.py CodeParrotSmall_agpl3_python_2023-03-27-21-21-29
