import chembl_downloader
import click
import pystow

SFI_MODULE = pystow.module("PatWalters", "sfi")

chembl_version_option = click.option("--chembl-version", default=chembl_downloader.latest)
