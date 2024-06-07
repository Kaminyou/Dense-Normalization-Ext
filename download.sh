wget -O materials_simple.zip https://www.dropbox.com/scl/fi/4m76ljmqxuhvko4gu6vs0/materials_simple.zip?rlkey=a2q4c69oq6uuuoyx3kv7045t5
EXPECTED_HASH="034cbbe468761fcbcb1b1a85058060aa9d59d3de9f1655510e29530c586637b3"
CALCULATED_HASH=$(sha256sum materials_simple.zip | awk '{ print $1 }')
# Compare the calculated hash to the expected hash
if [ "$CALCULATED_HASH" != "$EXPECTED_HASH" ]; then
  echo "Error: SHA-256 hash does not match"
else
  echo "SHA-256 hash matches"
  unzip materials_simple.zip
fi
