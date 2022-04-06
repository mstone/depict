terraform {
  required_providers {
    gandi = {
      source   = "go-gandi/gandi"
      version = "~> 2.0.0"
    }
    aws = {
      source = "hashicorp/aws"
      version = "~> 4.5.0"
    }
  }
}

resource "aws_instance" "diadym_server" {
  most_recent = true
  
}

data "gandi_domain" "diadym_com" {
  name = "diadym.com"
}

resource "gandi_livedns_domain" "diadym_com" {
  name = "diadym.com"
}

data "gandi_livedns_domain" "diadym_com" {
  name = "diadym.com"
}

resource "gandi_livedns_record" "at_diadym_com" {
  zone = "diadym.com"
  name = "@"
  type = "A"
  values = toset(["108.61.151.171"])
  ttl = "300"
}
