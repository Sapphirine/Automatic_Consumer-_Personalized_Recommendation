import twint
def get_timeline(username, limit,fn):
  c = twint.Config()
  c.Username = username
  c.pandas = True
  c.Output = fn
  c.Limit = limit
  c.Retweets = True
  c.Profile_full = True
  c.Lang = "en"
  twint.run.Profile(c)




