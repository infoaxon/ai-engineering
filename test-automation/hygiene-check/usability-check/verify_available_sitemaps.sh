for path in blogs-sitemap.xml journal-sitemap.xml kb-sitemap.xml wiki-sitemap.xml message-boards-sitemap.xml; do
  echo -n "$path: "
  curl -sk -o /dev/null -w "%{http_code}\n" "https://ppdolphin.brobotinsurance.com/$path"
done
